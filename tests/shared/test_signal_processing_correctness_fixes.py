"""Physical-power and independently normalized spectral regression controls."""

import numpy as np
import pytest
from scipy.signal import stft

from shared.sensing import signal_processing as s
from shared.sensing.detection.clutter import SlidingBaselineRemover


def test_range_and_rd_preserve_zero_and_weak_physical_power():
    profile = s.compute_range_profile(
        np.array([1, 1e-20]), np.array([0.0, 1.0]), sampling_frequency=1, fft_size=3, window="rect"
    )
    np.testing.assert_array_equal(profile.power_linear, [1, 1e-40, 0])
    samples = np.tile([1, 1e-4, 0], (16, 1)).astype(complex)
    rd = s.compute_range_doppler(
        fast_time_sequence=samples,
        sampling_frequency=1,
        prf_hz=100,
        fft_size_range=3,
        fft_size_doppler=16,
        window_range="rect",
        window_doppler="rect",
    )
    np.testing.assert_array_equal(rd.power_linear, abs(rd.cube) ** 2)
    assert rd.power_linear[8, 1] == pytest.approx(16e-8)
    assert rd.power_linear[0, 2] == 0


@pytest.mark.parametrize("perturbation", [0, 1e-10])
def test_truth_tracked_mti_has_no_constant_discontinuity(perturbation):
    n = 16
    times = np.arange(n) * 0.01
    truth = [
        {"reference_time_s": float(t), "target_ids": ["a"], "target_ranges": [0.0]} for t in times
    ]
    x = np.ones((n, 1), complex)
    x[-1] += perturbation
    result = s.compute_range_tracked_range_doppler(
        truth,
        target_id="a",
        fast_time_sequence=x,
        sample_times_s=times,
        range_axis_m=[0.0],
        fft_size_doppler=n,
        window_range="rect",
        window_doppler="rect",
        gate_half_width_bins=0,
        mti_enabled=True,
    )
    expected = (x - x[:1]) - (x - x[:1]).mean(axis=0)
    assert np.sum(result.power_linear) == pytest.approx(np.sum(abs(expected) ** 2), abs=1e-30)
    assert result.metadata["slow_time_mean_removal_policy"] == "window_weighted_mean"


@pytest.mark.parametrize("n", [31, 32])
@pytest.mark.parametrize("mti", [False, True])
@pytest.mark.parametrize("window", ["hann", "rect"])
def test_stft_matches_actual_scipy_with_explicit_normalization(n, mti, window):
    hop, fs = 8, 100.0
    t = np.arange(128) / fs
    x = np.stack([2 + np.exp(2j * np.pi * 12.5 * t), 0.7 * np.exp(-2j * np.pi * 18.75 * t)], axis=1)
    result = s.compute_range_collapsed_micro_doppler(
        fast_time_sequence=x,
        prf_hz=fs,
        frame_duration_s=1 / fs,
        stft_window_frames=n,
        stft_hop_frames=hop,
        window=window,
        mti_enabled=mti,
    )
    win = np.hanning(n) if window == "hann" else np.ones(n)
    win *= np.sqrt(n / np.sum(win**2))
    values = x - x.mean(axis=0) if mti else x
    freq, time, z = stft(
        values,
        fs=fs,
        window=win,
        nperseg=n,
        noverlap=n - hop,
        nfft=n,
        detrend=False,
        return_onesided=False,
        boundary=None,
        padded=False,
        axis=0,
        scaling="spectrum",
    )
    z *= win.sum() / np.sqrt(n)
    if mti:
        z[0] = 0
    reference = np.sum(abs(np.fft.fftshift(z, axes=0)) ** 2, axis=1).T
    np.testing.assert_allclose(result.spectrogram, reference, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(result.time_axis_s, time)
    np.testing.assert_array_equal(result.doppler_axis_hz, np.fft.fftshift(freq))


def test_noncoherent_range_integration_does_not_cancel_antiphase_channels():
    tone = np.exp(2j * np.pi * 5 * np.arange(32) / 32)
    p = s._compute_stft_power_spectrogram(
        np.stack([tone, -tone], axis=1),
        stft_window_frames=32,
        stft_hop_frames=8,
        window="rect",
        remove_dc=False,
    )
    assert p.sum() == pytest.approx(64)
    coherent = s.project_cir_to_delay_bins(
        np.array([1, -1]), np.array([0.0, 0.0]), sampling_frequency=1, fft_size=1
    )
    assert coherent[0] == 0


@pytest.mark.parametrize("n", [1, 2, 7, 8])
def test_analytical_nearest_fft_bin_and_empty_paths(n):
    result = s.compute_analytical_micro_doppler(
        [
            {"amplitudes": [1e-20], "doppler_hz": [0.9], "delays_s": [0.0]},
            {"amplitudes": [], "doppler_hz": [], "delays_s": []},
        ],
        prf_hz=8,
        n_doppler_bins=n,
    )
    axis = np.fft.fftshift(np.fft.fftfreq(n, d=1 / 8))
    np.testing.assert_array_equal(result.doppler_axis_hz, axis)
    circular_distance = abs((axis - 0.9 + 4) % 8 - 4)
    assert result.spectrogram[0].argmax() == circular_distance.argmin()
    assert result.spectrogram[0].sum() == pytest.approx(1e-40, abs=1e-50)
    assert not result.spectrogram[1].any()
    empty = s.compute_analytical_micro_doppler(
        [{"amplitudes": [], "doppler_hz": [], "delays_s": []}], n_doppler_bins=n
    )
    assert not empty.spectrogram.any() and np.isnan(empty.range_m)
    assert empty.truth_assisted


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_samples_are_rejected_without_zero_filling(value):
    with pytest.raises(ValueError):
        s._compute_stft_power_spectrogram(
            np.array([value]), stft_window_frames=32, stft_hop_frames=8, window="hann"
        )
    with pytest.raises(ValueError):
        s.compute_micro_doppler(
            0, fast_time_sequence=np.full((32, 1), value), prf_hz=10, frame_duration_s=0.1
        )
    with pytest.raises(ValueError):
        s.project_cir_to_delay_bins(
            np.array([1]), np.array([value]), sampling_frequency=1, fft_size=2
        )
    with pytest.raises(ValueError):
        SlidingBaselineRemover(2).process(np.array([value]))


@pytest.mark.parametrize("n", [1, 2])
def test_short_rectangular_windows_have_defined_normalization(n):
    p = s._compute_stft_power_spectrogram(
        np.ones(n), stft_window_frames=n, stft_hop_frames=1, window="rect", remove_dc=False
    )
    assert p.sum() == pytest.approx(n)
    if n == 2:
        with pytest.raises(ValueError, match="nonzero finite window"):
            s._compute_stft_power_spectrogram(
                np.ones(n), stft_window_frames=n, stft_hop_frames=1, window="hann"
            )


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_invalid_window_lengths_and_history_fail_closed(bad):
    with pytest.raises(ValueError):
        s._window("hann", bad)
    with pytest.raises(ValueError):
        SlidingBaselineRemover(bad)


def test_bad_baseline_sample_does_not_mutate_history():
    b = SlidingBaselineRemover(2)
    b.process(np.array([2.0]))
    with pytest.raises(ValueError):
        b.process(np.ones(2))
    result = b.process(np.array([3.0]))
    assert result.frames_used == 1 and result.cleaned[0] == 1


def test_masks_preserve_inputs_and_reject_invalid_power():
    x = np.ones((8, 2))
    y = s.apply_pre_cfar_doppler_guards(x, edge_guard_bins=1, dc_guard_bins=1)
    assert x.sum() == 16 and y.sum() == 10
    for bad in [np.array([1]), np.array([[-1.0]]), np.array([[np.nan]])]:
        with pytest.raises(ValueError):
            s.apply_pre_cfar_doppler_guards(bad)
