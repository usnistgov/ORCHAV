"""Raw coverage inspection and ray ordering independent of display saturation."""

from types import SimpleNamespace

import numpy as np

from visualizer.src.coverage.volume_inspection import describe_cell, scalar_template


def model():
    values = np.array([[[-100, -20], [-10, np.nan]], [[-np.inf, -50], [-60, -70]]], np.float32)
    values.flags.writeable = False
    return SimpleNamespace(
        values=values,
        axes=(np.array([12, 14]), np.array([20, 23]), np.array([30, 34])),
        quantity="path_gain_db",
        transmitter="tx_recorded",
        availability=np.array([[[1, 1], [1, 0]], [[2, 1], [1, 1]]], np.uint8),
    )


def test_scalar_template_preserves_above_scale_order_and_exact_finite_source():
    source = model()
    payload = scalar_template(source, -130, -50)
    mask = np.isfinite(source.values)
    np.testing.assert_array_equal(payload.values[mask], source.values[mask])
    assert payload.values[0, 1, 0] > payload.values[0, 0, 1]  # -10 > -20, both above colour maximum
    assert payload.values[1, 0, 0] < payload.values[mask].min()  # saved zero contribution
    assert payload.values[0, 1, 1] < payload.values[1, 0, 0]  # absent below zero contribution
    assert payload.palette[0, 3] == 0 and payload.palette[1, 3] == 0
    assert not source.values.flags.writeable


def test_inspection_uses_raw_db_not_palette_codes():
    text = describe_cell(model(), (0, 1, 0), "Ray-maximum hit")
    assert "-10 dB" in text and "X 12, Y 23, Z 30 m" in text
    assert "tx_recorded" in text and "positive saved estimate" in text


def test_missing_and_zero_contributions_are_never_sensing_misses():
    source = model()
    missing = describe_cell(source, (1, 1, 0), "Source slice cell")
    zero = describe_cell(source, (0, 0, 1), "Source slice cell")
    assert "unavailable (NaN)" in missing and "reason not recorded" in missing
    assert "-inf dB" in zero and "zero solver contribution" in zero
    assert "no sensing detection outcome or trial" in missing
    assert "not proof of no physical signal" in zero


def test_aggregate_never_invents_winning_transmitter():
    source = model()
    source.transmitter = "@strongest-all"
    assert "winning TX identity not retained" in describe_cell(source, (0, 0, 0), "MIP")
