"""Independent complex-sum qualification of the bounded CFR model (T01-T18)."""

import numpy as np
import pytest

from shared.sensing.range_response import (
    FrequencyNoise,
    RangeResponseConfig,
    frequency_to_delay,
    paths_to_frequency,
)


def project(
    delays=(3.3,), gains=(1 + 2j,), *, n=16, fft=16, window="boxcar", elements=1, variance=0.0
):
    cfg = RangeResponseConfig("cfr_ifft", n, window, FrequencyNoise(variance, 42))
    times = 4 + np.arange(8) / 1000
    a = np.broadcast_to(np.asarray(gains, complex)[None, :, None], (elements, len(gains), 8)).copy()
    tau = np.asarray(delays) / 100e6
    h, diag = paths_to_frequency(
        a, tau, np.ones(len(gains), bool), times, bandwidth=100e6, config=cfg
    )
    response, axes, meta = frequency_to_delay(
        h, bandwidth=100e6, config=cfg, fft_size=fft, range_mode="monostatic"
    )
    return a, tau, h, response, axes, meta, diag


@pytest.mark.parametrize("n,fft", [(15, 15), (15, 16), (15, 45), (16, 16), (16, 31), (16, 64)])
@pytest.mark.parametrize("window", ["boxcar", "hann"])
def test_T02_T03_T09_T10_T11_direct_complex_sum(n, fft, window):
    a, tau, observed, result, axes, meta, _ = project(n=n, fft=fft, window=window)
    q = np.arange(n) - n // 2
    weights = np.ones(n) if window == "boxcar" else np.sin(np.pi * np.arange(n) / n) ** 2
    expected_h = a[0, 0, 0] * np.exp(-2j * np.pi * (q * 100e6 / n) * tau[0])
    expected = np.array(
        [
            np.sum(weights * expected_h * np.exp(2j * np.pi * q * j / fft)) / weights.sum()
            for j in range(fft)
        ]
    )
    np.testing.assert_allclose(observed[0, 0], expected_h, atol=2e-14)
    np.testing.assert_allclose(result[0, 0], expected, atol=2e-14)
    assert meta["range_resolution_scale_m"] == pytest.approx(299792458 / 2 / 100e6)
    assert axes["delay_axis_s"][1] == pytest.approx(n / (100e6 * fft))
    on_grid = project((3,), n=n, fft=n, window=window)[3]
    assert on_grid[0, 0, 3] == pytest.approx(1 + 2j)


def test_T04_boundary_continuity():
    left = project((3 - 1e-7,))[3]
    right = project((3 + 1e-7,))[3]
    assert np.max(abs(left - right)) < 1e-6
    from shared.sensing.signal_processing import cir_to_samples_batch

    before = cir_to_samples_batch(
        np.ones((1, 8), complex),
        np.array([(3 - 1e-7) / 100e6]),
        sampling_frequency=100e6,
        fft_size=16,
    )
    after = cir_to_samples_batch(
        np.ones((1, 8), complex),
        np.array([(3 + 1e-7) / 100e6]),
        sampling_frequency=100e6,
        fft_size=16,
    )
    assert np.argmax(abs(before[0])) == 2
    assert np.argmax(abs(after[0])) == 3


@pytest.mark.parametrize("gains", [(1, 1), (1, -1), (1, 0.2j)])
def test_T05_T06_coherent_interference(gains):
    together = project((3.3, 3.3), gains)[3]
    first = project((3.3,), (gains[0],))[3]
    second = project((3.3,), (gains[1],))[3]
    np.testing.assert_allclose(together, first + second, atol=1e-15)
    unequal = project((3.3, 4.1), gains)[3]
    np.testing.assert_allclose(unequal, first + project((4.1,), (gains[1],))[3], atol=1e-15)


@pytest.mark.parametrize("tone", [-125, 125])
def test_T07_T08_T15_phase_and_element_order(tone):
    cfg = RangeResponseConfig("cfr_ifft", 16)
    times = 2 + np.arange(8) / 1000
    ratios = np.exp(1j * np.arange(4) * 0.7)
    a = ratios[:, None, None] * np.exp(2j * np.pi * tone * (times - 2))[None, None, :]
    h, _ = paths_to_frequency(
        a, np.array([3 / 100e6]), np.array([True]), times, bandwidth=100e6, config=cfg
    )
    delay, _, _ = frequency_to_delay(
        h, bandwidth=100e6, config=cfg, fft_size=16, range_mode="monostatic"
    )
    np.testing.assert_allclose(delay[:, :, 3], a[:, 0, :], atol=1e-14)
    rd = np.fft.fftshift(np.fft.fft(delay, axis=1), axes=1)
    axis = np.fft.fftshift(np.fft.fftfreq(8, 0.001))
    assert axis[np.argmax(abs(rd[0, :, 3]))] == tone
    # The primitive deliberately has no carrier parameter to rotate twice.


def test_T12_T13_absolute_coordinates_and_span():
    for d in (0, 0.00001, 15.99999):
        project((d,))
    for d in (-0.01, 16, 17):
        with pytest.raises(ValueError, match="ambiguity period"):
            project((d,))
    _, _, h, _, axes, _meta, _ = project((3.3,))
    _, bi, _ = frequency_to_delay(
        h,
        bandwidth=100e6,
        config=RangeResponseConfig("cfr_ifft", 16),
        fft_size=16,
        range_mode="bistatic",
    )
    np.testing.assert_array_equal(bi["range_axis_m"], 2 * axes["range_axis_m"])


def test_T14_empty_invalid_nonfinite_unavailable():
    cfg = RangeResponseConfig("cfr_ifft", 16)
    times = np.arange(4) / 1000
    h, _ = paths_to_frequency(
        np.empty((1, 0, 4), complex),
        np.empty(0),
        np.empty(0, bool),
        times,
        bandwidth=100e6,
        config=cfg,
    )
    assert not h.any()
    h, diag = paths_to_frequency(
        np.full((1, 2, 4), np.nan + 1j),
        np.array([-1, np.nan]),
        np.array([False, True]),
        times,
        bandwidth=100e6,
        config=cfg,
    )
    assert not h.any()
    assert diag["invalid_entry_count"] == 1
    assert diag["nonfinite_path_count"] == 1
    with pytest.raises(ValueError, match="Unavailable"):
        paths_to_frequency(None, None, None, times, bandwidth=100e6, config=cfg)


@pytest.mark.parametrize("window", ["boxcar", "hann"])
def test_T16_T17_noise_variance_and_correlation(window):
    cfg = RangeResponseConfig("cfr_ifft", 32, window, FrequencyNoise(2.0, 123))
    args = (np.empty((1, 0, 8192), complex), np.empty(0), np.empty(0, bool), np.arange(8192) / 1000)
    h, _ = paths_to_frequency(*args, bandwidth=100e6, config=cfg)
    identical, _ = paths_to_frequency(*args, bandwidth=100e6, config=cfg)
    np.testing.assert_array_equal(h, identical)
    result, _, meta = frequency_to_delay(
        h, bandwidth=100e6, config=cfg, fft_size=128, range_mode="monostatic"
    )
    assert np.mean(abs(result) ** 2) == pytest.approx(2 * meta["noise_gain"], rel=0.02)
    unpadded, _, _ = frequency_to_delay(
        h, bandwidth=100e6, config=cfg, fft_size=32, range_mode="monostatic"
    )
    np.testing.assert_allclose(result[..., ::4], unpadded, atol=1e-15)
    assert abs(np.mean(result[..., 1:] * result[..., :-1].conj())) > 0.8 * np.mean(abs(result) ** 2)


@pytest.mark.parametrize(
    "bad",
    [
        dict(num_frequency_bins=True),
        dict(num_frequency_bins=4.5),
        dict(frequency_window="hamming"),
        dict(frequency_noise={"variance": -1}),
        dict(frequency_noise={"variance": 1}),
    ],
)
def test_T18_invalid_configuration(bad):
    with pytest.raises(ValueError):
        RangeResponseConfig.parse({"model": "cfr_ifft", "num_frequency_bins": 16, **bad})


def test_T18_no_double_noise_or_delay_taper():
    cfg = RangeResponseConfig("cfr_ifft", 16)
    for overrides in (
        {"window_range": "hann"},
        {"noise_enabled": True},
        {"noise_snr_db": 10},
        {"processing_mode": "analytic_path_doppler_oracle"},
        {"fft_size": 8},
    ):
        with pytest.raises(ValueError):
            cfg.validate_processing(
                **{"bandwidth": 100e6, "fft_size": 16, "window_range": "boxcar", **overrides}
            )
