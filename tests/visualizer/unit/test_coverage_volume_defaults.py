"""Matched ordinary/volume views and transparent non-finite signal rendering."""

from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib import colormaps

from shared.coverage.volume import ALL_TRANSMITTERS
from visualizer.src.coverage.volume_defaults import initial_volume_controls
from visualizer.src.coverage.volume_inspection import scalar_template


@pytest.mark.parametrize("quantity", ["path_gain_db", "rss_dbm", "path_loss_db"])
def test_scalar_palette_keeps_weak_finite_values_and_hides_zero_and_missing(quantity):
    loss = quantity == "path_loss_db"
    values = np.array([[[np.nan, -np.inf, -150, -95, -70, -30]]], dtype=np.float32)
    if loss:
        values = -values
    model = SimpleNamespace(values=values, quantity=quantity)
    original = values.copy()
    low, high = (70, 100) if loss else (-100, -70)
    cmap = "RdYlGn_r" if loss else "RdYlGn"
    template = scalar_template(model, low, high, colormap=cmap)
    indices = np.rint(
        (template.values - template.limits[0]) / np.ptp(template.limits) * 1023
    ).astype(int)
    rgba = template.palette[indices][0, 0]
    np.testing.assert_equal(rgba[:2, 3], 0)
    np.testing.assert_equal(rgba[2:, 3], 1)
    # Below-scale finite powers and above-scale finite losses saturate, never disappear.
    np.testing.assert_allclose(rgba[2], colormaps[cmap](1.0 if loss else 0.0))
    np.testing.assert_allclose(rgba[-1], colormaps[cmap](0.0 if loss else 1.0))
    assert template.values[0, 0].argmax() == 5  # Physical strongest/lowest loss.
    np.testing.assert_equal(model.values, original)
    grey = scalar_template(model, low, high, colormap=cmap, show_nonfinite=True)
    np.testing.assert_equal(grey.values, template.values)
    np.testing.assert_equal(grey.palette[2:], template.palette[2:])
    np.testing.assert_equal(grey.palette[:2], [[0.5, 0.5, 0.5, 1]] * 2)


@pytest.mark.parametrize(
    "metric,quantity,tx,cmap",
    [
        ("best_path_loss_db", "path_loss_db", ALL_TRANSMITTERS, "RdYlGn_r"),
        ("best_rss_dbm", "rss_dbm", ALL_TRANSMITTERS, "RdYlGn"),
        ("rss_dbm/B", "rss_dbm", "B", "RdYlGn"),
        ("path_loss_db/1", "path_loss_db", "B", "RdYlGn_r"),
    ],
)
def test_initial_controls_preserve_active_selection(metric, quantity, tx, cmap):
    viz = SimpleNamespace(
        coverage_data=dict(
            tx_names=["A", "B"], metric_name=metric, value_min=55.123456, value_max=147.234567
        ),
        coverage_height_index=8,
        coverage_opacity=0.9,
    )
    controls = initial_volume_controls(viz)
    assert controls["transmitter"] == tx
    assert controls["quantity"] == quantity
    assert controls["colormap"] == cmap
    assert controls["height_index"] == 8
    assert controls["opacity"] == 0.9
    assert (controls["low"], controls["high"]) == (55.123456, 147.234567)


def test_unsupported_quantity_is_explained_instead_of_claimed_equivalent():
    controls = initial_volume_controls(
        SimpleNamespace(
            coverage_data=dict(tx_names=["A", "B"], metric_name="sinr_db/A"),
        )
    )
    assert "not available" in controls["note"]
    assert "sinr_db/A" in controls["note"]


def test_linear_scale_is_converted_to_equivalent_db():
    controls = initial_volume_controls(
        SimpleNamespace(
            coverage_data=dict(
                tx_names=["A"], metric_name="rss_w/A", value_min=1e-12, value_max=1e-6
            ),
        )
    )
    assert controls["quantity"] == "rss_dbm"
    np.testing.assert_allclose([controls["low"], controls["high"]], [-90, -30])
    assert "equivalent dB" in controls["note"]


def test_linear_power_volume_uses_logarithmic_colours_and_retains_order():
    model = SimpleNamespace(values=np.array([[[1e-12, 1e-9, 1e-6]]]), quantity="rss_w")
    template = scalar_template(model, 1e-12, 1e-6)
    np.testing.assert_allclose(template.values, [[[-12, -9, -6]]])
    assert template.values.argmax() == 2


def test_serving_regions_use_ids_without_averaging_or_relabelling_other_servers():
    from visualizer.src.coverage.analysis import serving_tx_color_rgb
    from visualizer.src.coverage.volume_inspection import categorical_template

    model = SimpleNamespace(values=np.array([[[0, 1, -1, np.nan]]]), transmitter_names=("A", "B"))
    template = categorical_template(model, show_nonfinite=True, transmitter_index=1)
    np.testing.assert_equal(template.values, [[[0, 3, 1, 1]]])
    np.testing.assert_equal(template.palette[0], [0, 0, 0, 0])
    np.testing.assert_allclose(template.palette[3, :3], serving_tx_color_rgb(1))
    np.testing.assert_equal(template.palette[1], [0.5, 0.5, 0.5, 1])
