"""Independent Fourier and absolute-phase checks, separate from GUI rendering."""

import numpy as np
import pytest

from scripts.analysis.validate_range_doppler import (
    BANDWIDTH,
    C0,
    CARRIER,
    FREQUENCIES,
    RANGE_FFT,
    SAMPLE_RATE,
    SAMPLES,
    WINDOWS,
    direct_response,
    observation_payload,
    point_observation,
    process,
)
from shared.sensing.range_response import frequency_grid


@pytest.mark.parametrize("rw,dw", WINDOWS)
@pytest.mark.parametrize("velocity", [0.0, 1.8, -1.8])
def test_complex_range_doppler_matches_explicit_double_fourier_sum(rw, dw, velocity):
    h, f, t = point_observation(velocity=velocity)
    result = process(observation_payload(h, f, t), rw, dw)
    delay, rd = direct_response(h, rw, dw)
    np.testing.assert_allclose(result["delay"][0], delay, atol=3e-14, rtol=3e-14)
    np.testing.assert_allclose(result["rd_cube"], rd, atol=2e-13, rtol=3e-13)
    np.testing.assert_allclose(result["range_profile"], abs(delay[-1]) ** 2, atol=2e-14)
    row, col = np.unravel_index(np.argmax(abs(rd)), rd.shape)
    speed = result["doppler_axis_hz"][row] * C0 / (2 * CARRIER)
    assert abs(speed - velocity) <= SAMPLE_RATE / SAMPLES * C0 / (4 * CARRIER)
    expected_mid = 5 - velocity * (t[0] + t[-1]) / 2
    assert abs(result["range_axis_m"][col] - expected_mid) <= C0 * FREQUENCIES / (
        4 * BANDWIDTH * RANGE_FFT
    )


def test_grid_endpoints_resolution_and_unambiguous_axes():
    h, f, t = point_observation()
    r = process(observation_payload(h, f, t))
    np.testing.assert_array_equal(f, frequency_grid(BANDWIDTH, FREQUENCIES))
    assert f[0] == -250e6 and f[-1] == 246093750
    assert f[-1] - f[0] == 496093750  # endpoint span != N*delta_f
    assert np.diff(f)[0] == 3906250
    assert C0 / CARRIER == pytest.approx(0.0107068735)
    assert r["metadata"]["range_resolution_scale_m"] == pytest.approx(0.299792458)
    assert np.diff(r["range_axis_m"])[0] == pytest.approx(0.0749481145)
    assert r["range_axis_m"][-1] == pytest.approx(38.2984865095)
    assert C0 / (2 * np.diff(f)[0]) == pytest.approx(38.373434624)
    assert np.diff(r["doppler_axis_hz"])[0] == 15.625
    assert r["doppler_axis_hz"][0] * C0 / (2 * CARRIER) == pytest.approx(-2.676718375)
    assert t[-1] - t[0] == 0.063 and SAMPLES / SAMPLE_RATE == 0.064


def test_dirichlet_sidelobes_nulls_and_periodicity_without_fft_reference():
    h, f, t = point_observation(velocity=0.0)
    result = process(observation_payload(h, f, t))
    x = np.arange(RANGE_FFT) / RANGE_FFT - 2 * 5 / C0 * (BANDWIDTH / FREQUENCIES)
    x = (x + 0.5) % 1 - 0.5
    kernel = np.exp(-1j * np.pi * x) * np.sinc(FREQUENCIES * x) / np.sinc(x)
    expected = np.exp(-4j * np.pi * CARRIER * 5 / C0) * kernel
    np.testing.assert_allclose(result["delay"][0, 0], expected, atol=2e-12, rtol=2e-12)
    period = C0 * FREQUENCIES / (2 * BANDWIDTH)
    ranges = np.array([0.0, 1.0, 19.0, 38.0, period, period + 1.0, period + 19.0, period + 38.0])
    direct = h[0] @ np.exp(4j * np.pi * f[:, None] * ranges / C0) / FREQUENCIES
    np.testing.assert_allclose(direct[:4], direct[4:], atol=3e-14)
    # Compare lobe envelopes, not individual bins that may fall near a null.
    assert result["range_profile"][-16:].max() > result["range_profile"][312:328].max()


def test_zero_padding_preserves_original_complex_samples_and_normalization():
    h, f, t = point_observation(amplitude=2.0)
    source = observation_payload(h, f, t)
    for window in ("boxcar", "hann"):
        unpadded = process(source, window, size=FREQUENCIES)
        padded = process(source, window, size=RANGE_FFT)
        np.testing.assert_allclose(padded["delay"][..., ::4], unpadded["delay"], atol=2e-15)
        np.testing.assert_allclose(
            padded["range_profile"][::4], unpadded["range_profile"], atol=3e-15
        )
        assert (
            padded["metadata"]["range_resolution_scale_m"]
            == unpadded["metadata"]["range_resolution_scale_m"]
        )
    h, f, t = point_observation(velocity=0.0, range0=16 * C0 / (2 * BANDWIDTH), amplitude=2.0)
    for window in ("boxcar", "hann"):
        result = process(observation_payload(h, f, t), window)
        assert result["range_profile"].max() == pytest.approx(4.0, abs=1e-12)


def test_off_grid_leakage_and_mean_removal_order_have_distinct_dc_responses():
    h, f, t = point_observation(frozen_delay=True)
    source = observation_payload(h, f, t)
    rect = process(source)
    hann = process(source, doppler_window="hann")
    cleaned = process(source, doppler_window="hann", clean=True)
    column = np.argmax(rect["range_profile"])

    def dc_ratio(result):
        p = abs(result["rd_cube"][:, column]) ** 2
        return 10 * np.log10(p[SAMPLES // 2] / p.max())

    assert dc_ratio(rect) == pytest.approx(-31.3322763973, abs=1e-8)
    assert dc_ratio(hann) < -90
    assert dc_ratio(cleaned) > -40
    w = np.sin(np.pi * np.arange(SAMPLES) / (SAMPLES - 1)) ** 2
    mean = hann["delay"][0].mean(axis=0)
    predicted = hann["rd_cube"][SAMPLES // 2] - mean * w.sum() / np.sqrt(np.sum(w * w))
    np.testing.assert_allclose(cleaned["rd_cube"][SAMPLES // 2], predicted, atol=2e-15)
    weighted = observation_payload(h, f, t, clutter="windowed_mean")
    assert (
        np.max(abs(process(weighted, doppler_window="hann", clean=True)["rd_cube"][SAMPLES // 2]))
        < 1e-14
    )


def test_on_grid_doppler_has_no_rectangular_leakage():
    velocity = 20 * SAMPLE_RATE / SAMPLES * C0 / (2 * CARRIER)
    h, f, t = point_observation(velocity=velocity, frozen_delay=True)
    rd = process(observation_payload(h, f, t))["rd_cube"]
    peak = np.argmax(np.max(abs(rd), axis=1))
    assert peak == SAMPLES // 2 + 20
    assert np.max(abs(np.delete(rd, peak, axis=0))) < 3e-11


def test_three_cpis_preserve_absolute_phase_and_remain_separate_transforms():
    chunks = [point_observation(cpi=i) for i in range(3)]
    h = np.concatenate([a for a, _, _ in chunks])
    f = chunks[0][1]
    t = np.arange(192) / SAMPLE_RATE
    independent = np.exp(-4j * np.pi * (CARRIER + f[None, :]) * (5 - 1.8 * t[:, None]) / C0)
    np.testing.assert_allclose(h, independent, rtol=0, atol=2e-12)
    expected_ratio = np.exp(4j * np.pi * (CARRIER + f) * 1.8 / (C0 * SAMPLE_RATE))
    for boundary in [63, 127]:
        np.testing.assert_allclose(h[boundary + 1] / h[boundary], expected_ratio, atol=5e-12)
    for a, f, times in chunks:
        result = process(observation_payload(a, f, times))
        assert result["rd_cube"].shape == (64, 512)
        assert np.diff(result["doppler_axis_hz"])[0] == 15.625
        np.testing.assert_allclose(result["rd_cube"], direct_response(a)[1], atol=2e-13)
    # The native frozen-delay approximation is explicitly different from R(t)
    # at every absolute frequency, although carrier Doppler is the same.
    frozen = point_observation(frozen_delay=True)[0]
    np.testing.assert_allclose(h[:64, FREQUENCIES // 2], frozen[:, FREQUENCIES // 2], atol=2e-12)
    assert np.max(abs(h[:64] - frozen)) > 0.5


def test_windows_reduce_sidelobes_broaden_lobes_and_use_explicit_gains():
    from scripts.analysis.validate_range_doppler import measurements

    h, f, t = point_observation()
    source = observation_payload(h, f, t)
    values = [measurements(process(source, rw, dw)) for rw, dw in WINDOWS]
    rect, range_hann, time_hann, both = values
    assert (
        range_hann["range_lobe"]["half_power_width"] > 1.5 * rect["range_lobe"]["half_power_width"]
    )
    assert range_hann["range_lobe"]["sampled_peak_sidelobe_db"] < -30
    assert (
        time_hann["doppler_lobe"]["half_power_width"]
        > 1.5 * rect["doppler_lobe"]["half_power_width"]
    )
    assert time_hann["doppler_lobe"]["sampled_peak_sidelobe_db"] < -30
    assert range_hann["far_range_peak_relative_db"] < rect["far_range_peak_relative_db"] - 30
    assert time_hann["zero_at_target_relative_db"] < rect["zero_at_target_relative_db"] - 50
    assert both["range_lobe"]["sampled_peak_sidelobe_db"] < -30
    h, f, t = point_observation(velocity=0.0, range0=16 * C0 / (2 * BANDWIDTH))
    source = observation_payload(h, f, t)
    rectangular = process(source)
    tapered = process(source, doppler_window="hann")
    assert np.max(abs(tapered["rd_cube"]) ** 2) / np.max(
        abs(rectangular["rd_cube"]) ** 2
    ) == pytest.approx(0.65625)


def test_point_bin_projection_changes_range_model_and_preserves_complex_doppler():
    from scripts.analysis.validate_range_doppler_targets import native_processing

    times = np.arange(SAMPLES) / SAMPLE_RATE
    coefficients = (2j * np.exp(4j * np.pi * CARRIER * 1.8 * times / C0))[None, :]
    delays = np.array([2 * 5 / C0])
    floor = native_processing(coefficients, delays, "point_bin_floor")
    cfr = native_processing(coefficients, delays, "cfr_ifft")
    expected = np.zeros((SAMPLES, RANGE_FFT), complex)
    expected[:, 16] = coefficients[0]
    np.testing.assert_array_equal(floor["slow_time_by_pair_rx_element"][0, 0], expected)
    assert np.count_nonzero(floor["range_profile_by_pair"][0]) == 1
    assert np.count_nonzero(cfr["range_profile_by_pair"][0]) == RANGE_FFT
    assert floor["range_axis_m"][16] == pytest.approx(4.796679328)
    assert np.diff(floor["range_axis_m"])[0] == pytest.approx(0.299792458)
    assert np.diff(cfr["range_axis_m"])[0] == pytest.approx(0.0749481145)
    # A single path has the same normalized Doppler spectrum under either
    # projection even though their range profiles are fundamentally different.
    spectra = []
    for s in (floor, cfr):
        power = abs(s["range_doppler_by_pair"][0]) ** 2
        spectra.append(power[:, np.argmax(power.max(axis=0))] / power.max())
    np.testing.assert_allclose(*spectra, atol=3e-15)
