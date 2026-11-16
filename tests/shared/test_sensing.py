import numpy as np
import pytest

from shared.sensing.signal_processing import (
    C0,
    DetectionVelocityDisplay,
    GTEntry,
    _window,
    apply_detection_velocity_metadata,
    auto_select_range_bin,
    bistatic_bisector_velocity_from_doppler_m_s,
    bistatic_path_closing_rate_m_s,
    cir_to_samples,
    compute_gt_entry,
    compute_micro_doppler,
    compute_range_collapsed_micro_doppler,
    compute_range_doppler,
    compute_range_profile,
    compute_range_tracked_micro_doppler,
    compute_range_tracked_range_doppler,
    detection_velocity_display,
    doppler_axis_to_velocity_m_s,
    doppler_to_monostatic_equivalent_velocity_m_s,
    doppler_to_path_closing_rate_m_s,
    doppler_to_velocity_m_s,
    extract_target_positions,
    format_detection_velocity_m_s,
    has_fresh_sensing_data,
    inject_target_relative_complex_awgn,
    normalize_range_mode,
    radial_velocity_axis,
)


def test_target_relative_complex_awgn_is_seeded_and_matches_requested_power():
    samples = np.ones(400_000, dtype=np.complex128) * (2.0 + 0.0j)
    first = inject_target_relative_complex_awgn(
        samples,
        10.0,
        rng=np.random.default_rng(20260828),
    )
    second = inject_target_relative_complex_awgn(
        samples,
        10.0,
        rng=np.random.default_rng(20260828),
    )

    np.testing.assert_array_equal(first.samples, second.samples)
    assert first.reference_signal_power == pytest.approx(4.0)
    assert first.expected_noise_power == pytest.approx(0.4)
    assert first.realized_noise_power == pytest.approx(0.4, rel=0.01)
    assert first.realized_snr_db == pytest.approx(10.0, abs=0.05)
    assert first.reference_sample_count == samples.size
    assert first.metadata()["physical_thermal_noise_model"] is False
    assert first.metadata()["scale_semantics"] == "relative_unitless"


def test_target_relative_complex_awgn_labels_zero_reference_as_not_applied():
    samples = np.zeros((2, 3), dtype=np.complex64)
    result = inject_target_relative_complex_awgn(
        samples,
        0.0,
        rng=np.random.default_rng(7),
    )

    np.testing.assert_array_equal(result.samples, samples)
    assert result.applied is False
    assert result.realized_snr_db is None
    assert result.disposition == "no_nonzero_signal_reference"
    assert result.metadata()["application_support"] == "all_input_samples"


@pytest.mark.parametrize(
    ("samples", "snr_db", "error", "message"),
    [
        (np.ones(4, dtype=np.float64), 10.0, TypeError, "complex floating"),
        (
            np.array([1.0 + 0.0j, np.nan + 0.0j]),
            10.0,
            ValueError,
            "input samples must be finite",
        ),
        (np.ones(4, dtype=np.complex128), np.inf, ValueError, "snr_db must be finite"),
    ],
)
def test_target_relative_complex_awgn_rejects_undefined_inputs(samples, snr_db, error, message):
    with pytest.raises(error, match=message):
        inject_target_relative_complex_awgn(samples, snr_db)


@pytest.mark.parametrize("snr_db", [1.0e308, -1.0e308])
def test_target_relative_complex_awgn_rejects_extreme_finite_snr(snr_db):
    with pytest.raises(ValueError, match="requested SNR is outside numeric range"):
        inject_target_relative_complex_awgn(
            np.ones(4, dtype=np.complex128),
            snr_db,
            rng=np.random.default_rng(1),
        )


def test_target_relative_complex_awgn_complex64_reports_large_finite_power_without_overflow():
    result = inject_target_relative_complex_awgn(
        np.ones(64, dtype=np.complex64),
        -700.0,
        rng=np.random.default_rng(11),
    )

    assert np.all(np.isfinite(result.samples.real))
    assert np.all(np.isfinite(result.samples.imag))
    assert np.isfinite(result.expected_noise_power)
    assert np.isfinite(result.realized_noise_power)
    assert np.isfinite(result.realized_snr_db)


def test_target_relative_complex_awgn_rejects_noise_that_overflows_input_dtype():
    with pytest.raises(ValueError, match="input dtype numeric range"):
        inject_target_relative_complex_awgn(
            np.ones(64, dtype=np.complex64),
            -800.0,
            rng=np.random.default_rng(12),
        )


@pytest.mark.parametrize("snr_db", [True, np.bool_(False)])
def test_target_relative_complex_awgn_rejects_boolean_snr(snr_db):
    with pytest.raises(ValueError, match="snr_db must be finite"):
        inject_target_relative_complex_awgn(np.ones(4, dtype=np.complex64), snr_db)


def test_window_generation():
    # Test valid windows
    n = 10
    assert np.allclose(_window("hann", n), np.hanning(n))
    assert np.allclose(_window("hanning", n), np.hanning(n))
    assert np.allclose(_window("hamming", n), np.hamming(n))
    assert np.allclose(_window("blackman", n), np.blackman(n))

    # Test rectangular
    assert np.allclose(_window("rect", n), np.ones(n))
    assert np.allclose(_window("rectangular", n), np.ones(n))
    assert np.allclose(_window("none", n), np.ones(n))

    # Test default (None → rectangular) and unknown (→ Hann fallback)
    assert np.allclose(_window(None, n), np.ones(n))
    assert np.allclose(_window("unknown", n), np.hanning(n))


def test_cir_to_samples_basic():
    # Single tap at index 10
    fft_size = 64
    fs = 1e9
    delay_s = 10.0 / fs
    cir = np.array([1.0 + 0j])
    delays = np.array([delay_s])

    samples = cir_to_samples(cir, delays, sampling_frequency=fs, fft_size=fft_size)

    assert list(samples.shape) == [fft_size]
    assert np.abs(samples[10]) == 1.0
    assert np.sum(np.abs(samples)) == 1.0


def test_cir_to_samples_out_of_bounds():
    fft_size = 64
    fs = 1e9
    # Tap way out of range
    cir = np.array([1.0 + 0j])
    delays = np.array([100.0 / fs])

    samples = cir_to_samples(cir, delays, sampling_frequency=fs, fft_size=fft_size)
    assert np.sum(np.abs(samples)) == 0.0


def test_compute_range_profile_bistatic_peak():
    fs = 1e9
    fft_size = 256
    target_dist = 3.0  # meters (path length)
    delay_s = target_dist / C0

    cir = np.array([1.0 + 0j])
    delays = np.array([delay_s])

    rp = compute_range_profile(
        cir, delays, sampling_frequency=fs, fft_size=fft_size, window=None, range_mode="bistatic"
    )

    assert rp.power_linear.shape == (fft_size,)
    assert rp.range_axis_m.shape == (fft_size,)
    peak_idx = int(np.argmax(rp.power_linear))
    peak_loc = rp.range_axis_m[peak_idx]
    assert abs(peak_loc - target_dist) < (C0 / fs)
    assert rp.range_axis_label == "Path Length (m)"


def test_compute_range_profile_monostatic_peak():
    fs = 1e9
    target_dist = 3.0
    delay = 2 * target_dist / C0  # round trip

    cir = np.array([1.0 + 0j])
    delays = np.array([delay])

    rp = compute_range_profile(
        cir,
        delays,
        sampling_frequency=fs,
        fft_size=256,
        window=None,
        range_mode="monostatic",
    )

    peak_idx = np.argmax(rp.power_linear)
    peak_loc = rp.range_axis_m[peak_idx]
    assert abs(peak_loc - target_dist) < (C0 / (2 * fs))
    assert rp.range_axis_label == "Range (m)"


def test_compute_range_doppler_range_mode_axis():
    fs = 1e9
    prf = 1e3
    cir = np.array([1.0 + 0j])
    delays = np.array([0.0])
    cir_seq = [(cir, delays) for _ in range(4)]

    rd_bi = compute_range_doppler(
        cir_seq,
        sampling_frequency=fs,
        prf_hz=prf,
        fft_size_range=4,
        fft_size_doppler=4,
        range_mode="bistatic",
    )
    rd_mono = compute_range_doppler(
        cir_seq,
        sampling_frequency=fs,
        prf_hz=prf,
        fft_size_range=4,
        fft_size_doppler=4,
        range_mode="monostatic",
    )

    assert rd_bi is not None and rd_mono is not None
    assert rd_bi.range_axis_m[1] == pytest.approx(C0 / fs)
    assert rd_mono.range_axis_m[1] == pytest.approx(C0 / (2 * fs))
    assert rd_bi.range_axis_label == "Path Length (m)"
    assert rd_mono.range_axis_label == "Range (m)"


def test_compute_range_doppler_accepts_fast_time_sequence():
    fs = 1e9
    prf = 1e3
    cir = np.array([1.0 + 0j])
    delays = np.array([0.0])
    cir_seq = [(cir, delays) for _ in range(4)]

    rd_from_cir = compute_range_doppler(
        cir_seq,
        sampling_frequency=fs,
        prf_hz=prf,
        fft_size_range=4,
        fft_size_doppler=4,
    )

    fast_time_sequence = [
        cir_to_samples(cir, delays, sampling_frequency=fs, fft_size=4) for _ in range(4)
    ]
    rd_from_samples = compute_range_doppler(
        None,
        fast_time_sequence=fast_time_sequence,
        sampling_frequency=fs,
        prf_hz=prf,
        fft_size_range=4,
        fft_size_doppler=4,
    )

    assert rd_from_cir is not None and rd_from_samples is not None
    np.testing.assert_allclose(rd_from_cir.power_linear, rd_from_samples.power_linear)


def test_compute_micro_doppler_for_selected_range_bin():
    n_frames = 64
    n_bins = 4
    fast = np.zeros((n_frames, n_bins), dtype=np.complex128)
    t = np.arange(n_frames)
    fast[:, 2] = np.exp(1j * 2 * np.pi * 0.125 * t)

    result = compute_micro_doppler(
        2,
        fast_time_sequence=fast,
        prf_hz=1000.0,
        frame_duration_s=0.001,
        stft_window_frames=32,
        stft_hop_frames=8,
        sampling_frequency=1e9,
        range_mode="monostatic",
    )

    assert result is not None
    assert result.collapsed is False
    assert result.range_bin == 2
    assert result.spectrogram.shape[1] == 32


def test_compute_range_collapsed_micro_doppler_sums_range_bin_power():
    n_frames = 64
    n_bins = 4
    fast = np.zeros((n_frames, n_bins), dtype=np.complex128)
    t = np.arange(n_frames)
    sig = np.exp(1j * 2 * np.pi * 0.125 * t)
    fast[:, 1] = sig
    fast[:, 2] = sig

    single = compute_micro_doppler(
        1,
        fast_time_sequence=fast,
        prf_hz=1000.0,
        frame_duration_s=0.001,
        stft_window_frames=32,
        stft_hop_frames=8,
    )
    collapsed = compute_range_collapsed_micro_doppler(
        fast_time_sequence=fast,
        prf_hz=1000.0,
        frame_duration_s=0.001,
        stft_window_frames=32,
        stft_hop_frames=8,
        range_bins=[1, 2],
    )

    assert single is not None and collapsed is not None
    assert collapsed.collapsed is True
    assert collapsed.range_bin == -1
    assert collapsed.range_bin_start == 1
    assert collapsed.range_bin_end == 2
    np.testing.assert_allclose(
        collapsed.spectrogram,
        2.0 * single.spectrogram,
        rtol=1e-6,
        atol=1e-12,
    )


def test_micro_doppler_preserves_weak_signal_power_and_exact_zeros():
    """An amplitude change must scale power, not reveal an artificial floor."""
    rows = np.zeros((64, 4), dtype=complex)
    rows[:, 1] = np.exp(2j * np.pi * np.arange(64) / 8)
    kwargs = dict(prf_hz=1000.0, frame_duration_s=0.001)
    strong = compute_range_collapsed_micro_doppler(fast_time_sequence=rows, **kwargs)
    weak = compute_range_collapsed_micro_doppler(fast_time_sequence=rows * 1e-10, **kwargs)
    zero = compute_range_collapsed_micro_doppler(fast_time_sequence=rows * 0.0, **kwargs)
    assert strong is not None and weak is not None and zero is not None
    np.testing.assert_allclose(weak.spectrogram, strong.spectrogram * 1e-20, atol=1e-40)
    np.testing.assert_array_equal(zero.spectrogram, 0.0)


def test_compute_range_tracked_micro_doppler_handles_range_migration():
    n_frames = 96
    n_bins = 16
    fs = 1e9
    prf = 1000.0
    fast = np.zeros((n_frames, n_bins), dtype=np.complex128)
    t = np.arange(n_frames)

    # Inject one moving scatterer: range bin drifts from 4 → 10 over time.
    moving_bins = np.rint(np.linspace(4, 10, n_frames)).astype(int)
    sig = np.exp(1j * 2 * np.pi * 0.125 * t)
    fast[np.arange(n_frames), moving_bins] = sig

    range_res = C0 / (2.0 * fs)
    range_axis_m = np.arange(n_bins, dtype=np.float64) * range_res
    sample_times_s = np.arange(n_frames, dtype=np.float64) / prf
    gt_history = [
        {
            "reference_time_s": float(sample_time_s),
            "target_ids": ["moving-target"],
            "target_ranges": [float(bin_idx) * range_res],
            "target_speeds": [0.0],
        }
        for bin_idx, sample_time_s in zip(moving_bins, sample_times_s, strict=True)
    ]

    fixed = compute_micro_doppler(
        7,
        fast_time_sequence=fast,
        prf_hz=prf,
        frame_duration_s=1.0 / prf,
        stft_window_frames=32,
        stft_hop_frames=8,
        sampling_frequency=fs,
        range_mode="monostatic",
    )
    tracked = compute_range_tracked_micro_doppler(
        gt_history,
        target_id="moving-target",
        fast_time_sequence=fast,
        sample_times_s=sample_times_s,
        range_axis_m=range_axis_m,
        stft_window_frames=32,
        stft_hop_frames=8,
    )

    assert fixed is not None and tracked is not None
    assert tracked.range_tracking is True
    assert tracked.tracked_range_bins is not None
    assert tracked.tracked_range_bins.min() == 4
    assert tracked.tracked_range_bins.max() == 10
    # Tracking should recover more total signal power than a single fixed bin.
    assert float(np.sum(tracked.spectrogram)) > float(np.sum(fixed.spectrogram))


def test_compute_range_tracked_micro_doppler_rejects_truth_count_expansion():
    block_size = 8
    tracked_bins_by_cpi = np.array([3, 6, 9], dtype=int)
    n_frames = block_size * len(tracked_bins_by_cpi)
    n_bins = 16
    fs = 1e9
    prf = 1000.0

    fast = np.zeros((n_frames, n_bins), dtype=np.complex128)
    pulse_bins = np.repeat(tracked_bins_by_cpi, block_size)
    t = np.arange(n_frames)
    fast[np.arange(n_frames), pulse_bins] = np.exp(1j * 2 * np.pi * 0.125 * t)

    range_res = C0 / (2.0 * fs)
    range_axis_m = np.arange(n_bins, dtype=np.float64) * range_res
    sample_times_s = np.arange(n_frames, dtype=np.float64) / prf
    gt_history = [
        {
            "reference_time_s": float(sample_times_s[index * block_size]),
            "target_ids": ["moving-target"],
            "target_ranges": [float(bin_idx) * range_res],
            "target_speeds": [0.0],
        }
        for index, bin_idx in enumerate(tracked_bins_by_cpi)
    ]

    with pytest.raises(ValueError, match="truth snapshot per physical sample"):
        compute_range_tracked_micro_doppler(
            gt_history,
            target_id="moving-target",
            fast_time_sequence=fast,
            sample_times_s=sample_times_s,
            range_axis_m=range_axis_m,
            stft_window_frames=16,
            stft_hop_frames=4,
        )


def test_compute_range_tracked_range_doppler_centers_target_gate():
    n_frames = 96
    n_bins = 16
    fs = 1e9
    prf = 1000.0
    gate_half = 2

    fast = np.zeros((n_frames, n_bins), dtype=np.complex128)
    moving_bins = np.rint(np.linspace(4, 10, n_frames)).astype(int)
    t = np.arange(n_frames)
    fast[np.arange(n_frames), moving_bins] = np.exp(1j * 2 * np.pi * 0.125 * t)

    range_res = C0 / (2.0 * fs)
    range_axis_m = np.arange(n_bins, dtype=np.float64) * range_res
    sample_times_s = np.arange(n_frames, dtype=np.float64) / prf
    gt_history = [
        {
            "reference_time_s": float(sample_time_s),
            "target_ids": ["distractor", "moving-target"],
            "target_ranges": [range_axis_m[1], float(bin_idx) * range_res],
            "target_speeds": [0.0, 0.0],
        }
        for bin_idx, sample_time_s in zip(moving_bins, sample_times_s, strict=True)
    ]

    tracked_rd = compute_range_tracked_range_doppler(
        gt_history,
        target_id="moving-target",
        fast_time_sequence=fast,
        sample_times_s=sample_times_s,
        range_axis_m=range_axis_m,
        fft_size_doppler=32,
        gate_half_width_bins=gate_half,
    )

    assert tracked_rd is not None
    assert tracked_rd.power_linear.shape == (32, 2 * gate_half + 1)
    assert tracked_rd.range_axis_label == "Range Offset From Tracked Target (m)"
    peak_doppler_bin, peak_range_bin = np.unravel_index(
        np.argmax(tracked_rd.power_linear),
        tracked_rd.power_linear.shape,
    )
    assert peak_range_bin == gate_half
    assert tracked_rd.metadata["target_id"] == "moving-target"


def test_truth_tracking_rejects_unaligned_timestamp_and_missing_target_id():
    fast = np.ones((4, 3), dtype=np.complex128)
    range_axis_m = np.array([0.0, 0.75, 2.0], dtype=np.float64)
    sample_times_s = np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float64)
    truth = [
        {
            "reference_time_s": float(time_s),
            "target_ids": ["target-a", "target-b"],
            "target_ranges": [0.75, 2.0],
            "target_speeds": [0.0, 0.0],
        }
        for time_s in sample_times_s
    ]

    with pytest.raises(ValueError, match="explicit target_id"):
        compute_range_tracked_range_doppler(
            truth,
            target_id="",
            fast_time_sequence=fast,
            sample_times_s=sample_times_s,
            range_axis_m=range_axis_m,
        )

    truth[2] = {**truth[2], "reference_time_s": 1.25}
    with pytest.raises(ValueError, match="not exactly time-aligned"):
        compute_range_tracked_range_doppler(
            truth,
            target_id="target-b",
            fast_time_sequence=fast,
            sample_times_s=sample_times_s,
            range_axis_m=range_axis_m,
        )


# ---------------------------------------------------------------------------
# Tests for GT computation utilities
# ---------------------------------------------------------------------------


class TestExtractTargetPositions:
    def test_target_positions_snapshot(self):
        frame = {"target_positions_snapshot": [[1, 2, 3], [4, 5, 6]]}
        result = extract_target_positions(frame)
        assert result is not None
        assert len(result) == 2

    def test_target_pos_key(self):
        frame = {"target_pos": [[1, 2, 3]]}
        result = extract_target_positions(frame)
        assert result is not None
        assert len(result) == 1

    def test_no_targets(self):
        result = extract_target_positions({})
        assert result is None


class TestComputeGTEntry:
    def test_bistatic_range(self):
        tx = np.array([0.0, 0.0, 0.0])
        rx = np.array([0.0, 0.0, 0.0])
        target = [np.array([3.0, 0.0, 0.0])]
        entry = compute_gt_entry(tx, rx, target, range_mode="bistatic")
        # Bistatic path length = dist_tx + dist_rx = 3 + 3 = 6
        assert entry.target_ranges[0] == pytest.approx(6.0)
        assert entry.target_speeds[0] is None
        assert entry.target_velocity_valid == [False]
        assert entry.target_velocity_sources == ["unavailable"]
        assert entry.target_velocity_unavailable_reasons == [
            "no_prior_reference_epoch_or_target_velocity"
        ]

    def test_monostatic_range(self):
        tx = np.array([0.0, 0.0, 0.0])
        rx = np.array([0.0, 0.0, 0.0])
        target = [np.array([3.0, 0.0, 0.0])]
        entry = compute_gt_entry(tx, rx, target, range_mode="monostatic")
        # Monostatic range = 0.5 * (3 + 3) = 3
        assert entry.target_ranges[0] == pytest.approx(3.0)

    def test_speed_positive_approaching(self):
        """Positive speed = approaching (radar convention)."""
        tx = np.array([0.0, 0.0, 0.0])
        rx = np.array([0.0, 0.0, 0.0])
        # Target at 5m, previously at 6m → range decreased → approaching
        target = [np.array([5.0, 0.0, 0.0])]
        prev_ranges = [12.0]  # bistatic: was 12m (6+6)
        entry = compute_gt_entry(
            tx, rx, target, range_mode="bistatic", dt=1.0, prev_target_ranges=prev_ranges
        )
        # The full path closes at 2 m/s. The Doppler-comparable velocity is
        # the target-independent monostatic-equivalent value q/2 = 1 m/s.
        assert entry.target_ranges[0] == pytest.approx(10.0)
        assert entry.target_path_closing_rates_m_s[0] == pytest.approx(2.0)
        assert entry.target_speeds[0] == pytest.approx(1.0)
        assert entry.target_monostatic_equivalent_velocities_m_s[0] == pytest.approx(1.0)
        assert entry.target_velocity_valid == [True]
        assert entry.target_velocity_sources == ["exact_epoch_finite_difference"]
        assert entry.target_velocity_reference_dt_s == [1.0]

    def test_speed_negative_receding(self):
        """Negative speed = receding."""
        tx = np.array([0.0, 0.0, 0.0])
        rx = np.array([0.0, 0.0, 0.0])
        target = [np.array([5.0, 0.0, 0.0])]
        prev_ranges = [8.0]  # bistatic: was 8m
        entry = compute_gt_entry(
            tx, rx, target, range_mode="bistatic", dt=1.0, prev_target_ranges=prev_ranges
        )
        # Current = 10.0, prev = 8.0: q=-2 m/s and q/2=-1 m/s.
        assert entry.target_path_closing_rates_m_s[0] == pytest.approx(-2.0)
        assert entry.target_speeds[0] == pytest.approx(-1.0)

    def test_no_prev_range_and_no_velocity_is_explicitly_unavailable(self):
        tx = np.array([0.0, 0.0, 0.0])
        rx = np.array([0.0, 0.0, 0.0])
        target = [np.array([3.0, 0.0, 0.0])]
        entry = compute_gt_entry(tx, rx, target, range_mode="bistatic", dt=0.1)
        assert entry.target_speeds[0] is None
        assert entry.target_velocity_valid == [False]
        assert entry.target_velocity_sources == ["unavailable"]
        assert entry.target_velocity_reference_dt_s == [None]

    def test_monostatic_analytic_velocity_projection(self):
        tx = np.zeros(3)
        rx = np.zeros(3)
        target = [np.array([10.0, 0.0, 0.0])]
        entry = compute_gt_entry(
            tx,
            rx,
            target,
            range_mode="monostatic",
            target_velocities=np.array([[-2.0, 0.0, 0.0]]),
            target_ids=["walker"],
            reference_time_s=7.25,
        )

        assert entry.target_ids == ["walker"]
        assert entry.reference_time_s == pytest.approx(7.25)
        assert entry.target_path_closing_rates_m_s == pytest.approx([4.0])
        assert entry.target_speeds == pytest.approx([2.0])
        assert entry.target_bistatic_bisector_velocities_m_s == pytest.approx([2.0])
        assert entry.target_velocity_sources == ["analytic_velocity_projection"]
        assert entry.target_velocity_reference_dt_s == [None]

    def test_irregular_reference_epoch_finite_difference(self):
        entry = compute_gt_entry(
            np.zeros(3),
            np.zeros(3),
            [np.array([4.25, 0.0, 0.0])],
            range_mode="bistatic",
            dt=0.4,
            prev_target_ranges=[10.0],
            target_ids=["walker"],
            reference_time_s=3.4,
        )

        # Full bistatic path changed from 10 m to 8.5 m over exactly 0.4 s.
        assert entry.target_path_closing_rates_m_s == pytest.approx([3.75])
        assert entry.target_speeds == pytest.approx([1.875])
        assert entry.target_velocity_sources == ["exact_epoch_finite_difference"]
        assert entry.target_velocity_reference_dt_s == pytest.approx([0.4])

    def test_bistatic_analytic_speed_uses_bisector_projection(self):
        tx = np.array([-1.0, 0.0, 0.0])
        rx = np.array([1.0, 0.0, 0.0])
        target = [np.array([0.0, 1.0, 0.0])]
        velocities = np.array([[0.0, -2.0, 0.0]])

        entry = compute_gt_entry(
            tx,
            rx,
            target,
            range_mode="bistatic",
            dt=0.1,
            target_velocities=velocities,
        )

        assert entry.target_path_closing_rates_m_s[0] == pytest.approx(2.0 * np.sqrt(2.0))
        assert entry.target_speeds[0] == pytest.approx(np.sqrt(2.0))
        assert entry.target_bistatic_bisector_velocities_m_s[0] == pytest.approx(2.0)
        assert entry.target_velocity_valid == [True]
        assert entry.target_velocity_sources == ["analytic_velocity_projection"]

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"dt": 0.0, "prev_target_ranges": [1.0]}, "strictly positive"),
            ({"dt": np.nan, "prev_target_ranges": [1.0]}, "strictly positive"),
            ({"dt": True, "prev_target_ranges": [1.0]}, "strictly positive"),
            ({"dt": None, "prev_target_ranges": [1.0]}, "dt is required"),
            ({"reference_time_s": np.bool_(True)}, "reference_time_s must be finite"),
            ({"target_velocities": np.array([[np.nan, 0.0, 0.0]])}, "finite array"),
            ({"target_ids": ["alpha", "bravo"]}, "align one-to-one"),
        ],
    )
    def test_rejects_non_exact_epoch_velocity_and_identity_inputs(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            compute_gt_entry(
                np.zeros(3),
                np.zeros(3),
                [np.array([2.0, 0.0, 0.0])],
                **kwargs,
            )

    @pytest.mark.parametrize(
        "field, value",
        [
            ("tx", np.array([np.nan, 0.0, 0.0])),
            ("rx", np.array([0.0, np.inf, 0.0])),
            ("target", np.array([0.0, 0.0])),
        ],
    )
    def test_rejects_non_finite_or_malformed_positions(self, field, value):
        tx = value if field == "tx" else np.zeros(3)
        rx = value if field == "rx" else np.zeros(3)
        target = [value if field == "target" else np.ones(3)]
        with pytest.raises(ValueError, match="finite vector"):
            compute_gt_entry(tx, rx, target)

    def test_detections_extracted(self):
        tx = np.zeros(3)
        rx = np.zeros(3)
        detections = [
            {"range_m": 5.0, "velocity_m_s": 1.2},
            {"range_m": 8.0, "velocity_m_s": -0.5},
        ]
        entry = compute_gt_entry(tx, rx, [], detections=detections)
        assert entry.det_ranges == [5.0, 8.0]
        assert entry.det_velocities == [1.2, -0.5]
        assert entry.target_ranges == []

    def test_empty_targets(self):
        entry = compute_gt_entry(np.zeros(3), np.zeros(3), [])
        assert entry.target_ranges == []
        assert entry.target_speeds == []

    def test_multiple_targets(self):
        tx = np.array([0.0, 0.0, 0.0])
        rx = np.array([10.0, 0.0, 0.0])
        targets = [np.array([3.0, 0.0, 0.0]), np.array([7.0, 0.0, 0.0])]
        entry = compute_gt_entry(tx, rx, targets, range_mode="bistatic")
        # Target 0: dist_tx=3, dist_rx=7, path=10
        # Target 1: dist_tx=7, dist_rx=3, path=10
        assert len(entry.target_ranges) == 2
        assert entry.target_ranges[0] == pytest.approx(10.0)
        assert entry.target_ranges[1] == pytest.approx(10.0)

    def test_gt_entry_dataclass(self):
        entry = GTEntry(
            target_ranges=[1.0, 2.0],
            target_speeds=[0.5, -0.3],
            det_ranges=[1.1],
            det_velocities=[0.4],
        )
        assert entry.target_ranges == [1.0, 2.0]
        assert entry.det_ranges == [1.1]


def test_radial_velocity_axis_bistatic_uses_bisector():
    tx = np.array([-1.0, 0.0, 0.0])
    rx = np.array([1.0, 0.0, 0.0])
    target = np.array([0.0, 1.0, 0.0])

    axis = radial_velocity_axis(tx, rx, target, range_mode="bistatic")
    np.testing.assert_allclose(axis, np.array([0.0, -1.0, 0.0]), atol=1e-6)


def test_radial_velocity_axis_rejects_forward_scatter_singularity():
    with pytest.raises(ValueError, match="unobservable near forward scatter"):
        radial_velocity_axis(
            np.array([-1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.zeros(3),
            range_mode="bistatic",
        )


def test_doppler_observables_are_target_independent():
    assert doppler_to_path_closing_rate_m_s(2.0, 1.0) == pytest.approx(2.0)
    assert doppler_to_monostatic_equivalent_velocity_m_s(2.0, 1.0) == pytest.approx(1.0)
    assert doppler_to_velocity_m_s(2.0, 1.0) == pytest.approx(1.0)


def test_bistatic_bisector_velocity_is_evaluation_only_geometry_conversion():
    tx = np.array([-1.0, 0.0, 0.0])
    rx = np.array([1.0, 0.0, 0.0])
    target = np.array([0.0, 1.0, 0.0])

    velocity = bistatic_bisector_velocity_from_doppler_m_s(
        2.0,
        1.0,
        tx,
        rx,
        target,
    )

    assert velocity == pytest.approx(np.sqrt(2.0))


def test_bistatic_path_closing_rate_includes_moving_sensors():
    tx = np.array([-1.0, 0.0, 0.0])
    rx = np.array([1.0, 0.0, 0.0])
    target = np.array([0.0, 1.0, 0.0])
    target_velocity = np.array([0.0, -2.0, 0.0])
    tx_velocity = np.array([0.0, 0.5, 0.0])
    rx_velocity = np.array([0.0, 0.5, 0.0])

    closing_rate = bistatic_path_closing_rate_m_s(
        tx,
        rx,
        target,
        target_velocity,
        tx_velocity_m_s=tx_velocity,
        rx_velocity_m_s=rx_velocity,
    )

    assert closing_rate == pytest.approx(2.5 * np.sqrt(2.0))
    recovered = bistatic_bisector_velocity_from_doppler_m_s(
        closing_rate,
        1.0,
        tx,
        rx,
        target,
        tx_velocity_m_s=tx_velocity,
        rx_velocity_m_s=rx_velocity,
    )
    assert recovered == pytest.approx(2.0)


def test_bistatic_bisector_velocity_rejects_forward_scatter_singularity():
    with pytest.raises(ValueError, match="unobservable near forward scatter"):
        bistatic_bisector_velocity_from_doppler_m_s(
            1.0,
            1.0,
            np.array([-1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.zeros(3),
        )


@pytest.mark.parametrize("wavelength", [0.0, -1.0, float("nan"), float("inf")])
def test_doppler_conversion_rejects_nonphysical_wavelength(wavelength):
    with pytest.raises(ValueError, match="wavelength_m"):
        doppler_to_path_closing_rate_m_s(1.0, wavelength)


def test_detection_velocity_display_marks_bistatic_detections_as_approximate():
    display = detection_velocity_display(range_mode="bistatic")

    assert isinstance(display, DetectionVelocityDisplay)
    assert display.semantics == "monostatic_equivalent_velocity"
    assert display.axis_label == "Monostatic-equivalent Velocity (m/s)"
    assert display.is_approximate is False
    assert display.inline_suffix == " mono. equiv."


def test_format_detection_velocity_uses_approximate_suffix_for_bistatic():
    assert format_detection_velocity_m_s(1.25, range_mode="monostatic") == "1.2 m/s"
    assert format_detection_velocity_m_s(1.25, range_mode="bistatic") == "1.2 m/s mono. equiv."


def test_apply_detection_velocity_metadata_populates_mapping():
    payload: dict[str, object] = {}
    apply_detection_velocity_metadata(payload, range_mode="bistatic")

    assert payload["detection_velocity_semantics"] == "monostatic_equivalent_velocity"
    assert payload["detection_velocity_axis_label"] == "Monostatic-equivalent Velocity (m/s)"
    assert payload["detection_velocity_is_approximate"] is False
    assert payload["doppler_sign_convention"] == "positive_approaching"
    assert payload["path_closing_rate_formula"] == "wavelength_m * doppler_hz"
    assert payload["detection_velocity_formula"] == "0.5 * wavelength_m * doppler_hz"
    assert payload["detection_velocity_geometry_source"] == "none"


def test_doppler_axis_to_velocity_axis_matches_monostatic_conversion():
    doppler_axis = np.array([-2.0, 0.0, 2.0], dtype=np.float64)
    velocity_axis = doppler_axis_to_velocity_m_s(doppler_axis, carrier_frequency_hz=C0)

    np.testing.assert_allclose(velocity_axis, np.array([-1.0, 0.0, 1.0]))


def test_normalize_range_mode_defaults_to_bistatic():
    assert normalize_range_mode("monostatic") == "monostatic"
    assert normalize_range_mode("mono") == "monostatic"
    assert normalize_range_mode("bistatic") == "bistatic"
    assert normalize_range_mode(None) == "bistatic"


class TestAutoSelectRangeBin:
    def test_blind_default_does_not_use_gt_entries(self):
        gt_entries = [
            GTEntry(target_ranges=[5.0], target_speeds=[0.0], det_ranges=[], det_velocities=[]),
            GTEntry(target_ranges=[5.2], target_speeds=[0.1], det_ranges=[], det_velocities=[]),
        ]
        with pytest.raises(ValueError, match="center-bin and truth fallbacks are disabled"):
            auto_select_range_bin(gt_entries, {}, 128, 1e9, "bistatic")

    def test_truth_assisted_switch_is_rejected(self):
        gt_entries = [{"target_ranges": [3.0]}, {"target_ranges": [3.5]}]
        with pytest.raises(ValueError, match="blind-only"):
            auto_select_range_bin(
                gt_entries,
                {},
                128,
                1e9,
                "bistatic",
                truth_assisted=True,
            )

    def test_fallback_detection(self):
        sensing_data = {"detections": [{"range_m": 4.0, "power_linear": 1.0}]}
        bin_idx = auto_select_range_bin([], sensing_data, 128, 1e9, "bistatic")
        assert 0 <= bin_idx < 128

    def test_fallback_centre(self):
        with pytest.raises(ValueError, match="center-bin and truth fallbacks are disabled"):
            auto_select_range_bin([], {}, 128, 1e9, "bistatic")


class TestHasFreshSensingData:
    def test_empty_block_is_not_fresh(self):
        assert has_fresh_sensing_data({}) is False
        assert has_fresh_sensing_data(None) is False

    def test_temporal_result_marks_fresh_even_with_no_detections(self):
        sensing = {
            "schema_version": 4,
            "temporal": {"result": {"reference_time_s": 1.0}},
            "detections": [],
        }
        assert has_fresh_sensing_data(sensing) is True

    def test_acquisition_without_result_is_not_fresh(self):
        sensing = {"schema_version": 4, "temporal": {"acquisition": {}}}
        assert has_fresh_sensing_data(sensing) is False

    def test_legacy_array_presence_is_not_interpreted(self):
        from shared.extensions.sensing import UnsupportedSensingSchemaError

        with pytest.raises(UnsupportedSensingSchemaError, match="Regenerate"):
            has_fresh_sensing_data({"range_doppler_map": np.zeros((2, 4))})

    def test_malformed_temporal_result_fails(self):
        with pytest.raises(ValueError, match="temporal.result"):
            has_fresh_sensing_data({"schema_version": 4, "temporal": {"result": []}})
