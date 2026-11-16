"""CPU numerical regressions against explicit Fourier sums and window theory."""

import numpy as np
import pytest

from scripts.analysis.validate_range_doppler import (
    C0,
    CARRIER,
    FREQUENCIES,
    SAMPLES,
    WINDOWS,
    direct_response,
    observation_payload,
    process,
)
from scripts.analysis.validate_sensing_hann import (
    AMPLITUDE,
    CASES,
    RANGE_SCALE,
    VELOCITY_SCALE,
    dense_profiles,
    detect,
    match_detections,
    synthetic,
)
from shared.sensing.range_response import frequency_weights


@pytest.mark.parametrize("velocity", [0.0, 1.0, -1.0])
@pytest.mark.parametrize("rw,dw", WINDOWS)
def test_point_range_velocity_and_independent_fourier_sum(velocity, rw, dw):
    source = synthetic([(5.0, velocity, AMPLITUDE)])
    result = process(source, rw, dw)
    delay, rd = direct_response(source["observation"][0, 0], rw, dw)
    np.testing.assert_allclose(result["delay"][0], delay, atol=3e-13)
    np.testing.assert_allclose(result["rd_cube"], rd, atol=3e-12)
    i, j = np.unravel_index(abs(rd).argmax(), rd.shape)
    assert abs(result["range_axis_m"][j] - 5) <= RANGE_SCALE / 8
    assert abs(result["doppler_axis_hz"][i] * C0 / (2 * CARRIER) - velocity) <= VELOCITY_SCALE / 2
    np.testing.assert_allclose(result["range_profile"], abs(delay[-1]) ** 2, atol=1e-13)


@pytest.mark.parametrize("rw,dw", WINDOWS)
def test_coherent_gain_energy_and_padding(rw, dw):
    source = synthetic([(16 * RANGE_SCALE, 12 * VELOCITY_SCALE, AMPLITUDE)])
    result = process(source, rw, dw)
    w = np.hanning(SAMPLES) if dw == "hann" else np.ones(SAMPLES)
    gain = w.sum() ** 2 / np.sum(w * w)
    power = abs(result["rd_cube"]) ** 2
    assert power.max() == pytest.approx(abs(AMPLITUDE) ** 2 * gain, rel=1e-12)
    assert result["range_profile"].max() == pytest.approx(abs(AMPLITUDE) ** 2, rel=1e-12)
    assert power[:, 64].sum() == pytest.approx(SAMPLES * abs(AMPLITUDE) ** 2, rel=1e-12)
    unpadded = process(source, rw, dw, size=FREQUENCIES)
    np.testing.assert_allclose(result["delay"][..., ::4], unpadded["delay"], atol=1e-14)
    wr = frequency_weights(FREQUENCIES, rw)
    assert result["metadata"]["coherent_gain"] == pytest.approx(wr.sum())
    assert result["metadata"]["noise_gain"] == pytest.approx(
        (1.5 if rw == "hann" else 1) / FREQUENCIES
    )


@pytest.fixture(scope="module")
def measured_windows():
    source = synthetic(CASES["stationary"])
    return {(rw, dw): dense_profiles(source, rw, dw) for rw, dw in WINDOWS}


@pytest.mark.parametrize("rw,dw", WINDOWS)
def test_sidelobes_mainlobe_widths_and_dimension_independence(measured_windows, rw, dw):
    result = measured_windows[rw, dw]
    for dimension, window in (("range", rw), ("doppler", dw)):
        metric = result[dimension]
        assert metric["psl_db"] == pytest.approx(-31.47 if window == "hann" else -13.26, abs=0.04)
        assert metric["isl_db"] == pytest.approx(-32.88 if window == "hann" else -9.68, abs=0.04)
    np.testing.assert_allclose(
        result["range_power"], measured_windows[rw, "boxcar"]["range_power"], atol=1e-14
    )
    ratio_range = (
        measured_windows["hann", dw]["range"]["half_power_width"]
        / measured_windows["boxcar", dw]["range"]["half_power_width"]
    )
    ratio_doppler = (
        measured_windows[rw, "hann"]["doppler"]["half_power_width"]
        / measured_windows[rw, "boxcar"]["doppler"]["half_power_width"]
    )
    assert ratio_range == pytest.approx(1.626, abs=0.005)
    assert ratio_doppler == pytest.approx(1.652, abs=0.005)


@pytest.mark.parametrize("rw,dw", WINDOWS)
def test_stationary_stripe_matches_last_sample_up_to_coherent_gain(rw, dw):
    result = process(synthetic(CASES["stationary"]), rw, dw)
    w = np.hanning(SAMPLES) if dw == "hann" else np.ones(SAMPLES)
    expected = result["range_profile"] * w.sum() ** 2 / np.sum(w * w)
    np.testing.assert_allclose(abs(result["rd_cube"][SAMPLES // 2]) ** 2, expected, atol=1e-12)


@pytest.mark.parametrize("rw,dw", WINDOWS)
def test_separated_targets_recovered_without_truth_in_detector(rw, dw):
    result = process(synthetic(CASES["two_separated"]), rw, dw)
    score = match_detections(detect(result), CASES["two_separated"])
    assert score["tp"] == 2
    assert score["fn"] == 0
    if rw == "hann":
        assert score["fp"] == 0


def test_weak_target_contrast_improves_but_cfar_training_contamination_remains():
    strong = synthetic(CASES["strong_weak"][:1])
    responses = [process(strong, rw, "hann") for rw in ("boxcar", "hann")]
    j = int(np.argmin(abs(responses[0]["range_axis_m"] - 6.05)))
    leakage = [r["range_profile"][j] for r in responses]
    assert 10 * np.log10(leakage[1] / leakage[0]) < -15
    # A lower sidelobe is not a guarantee of CA-CFAR detection: its reference
    # cells still include the strong target. Record the actual limitation.
    result = process(synthetic(CASES["strong_weak"]), "hann", "hann")
    assert match_detections(detect(result), CASES["strong_weak"])["fn"] == 1


@pytest.mark.parametrize("rw,dw", WINDOWS)
def test_zero_power_remains_zero_without_artificial_scientific_floor(rw, dw):
    result = process(synthetic([(5.0, 0.0, 0.0)]), rw, dw)
    assert not np.any(result["rd_cube"])
    assert not np.any(result["range_profile"])
    assert detect(result) == []


def test_two_sample_hann_replay_rejects_zero_energy_window():
    source = synthetic(CASES["stationary"])
    short = observation_payload(
        source["observation"][0, 0, :2],
        source["frequency_axis_hz"],
        source["slow_time_axis_s"][:2],
        clutter="none",
    )
    with pytest.raises(ValueError, match="zero energy"):
        process(short, "boxcar", "hann")
    assert process(short, "boxcar", "boxcar")["range_profile"].max() > 0
