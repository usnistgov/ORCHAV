"""Independent Fourier and persisted-membership checks for sequential SISO CFR."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from generator.core.configuration import SensingConfig
from generator.core.exceptions import ConfigurationError
from generator.core.pipeline.schedule import FixedIntervalScheduler
from generator.core.pipeline.sensing import SensingPipelineState
from generator.core.sensing.acquisition import resolve_sensing_execution_values
from generator.core.sensing.processor import create_sensing_processor
from shared.sensing.cfr_replay import rebuild_frequency_observation
from shared.sensing.postprocessor import PostProcessingConfig, SensingPostProcessor
from shared.sensing.range_response import (
    FrequencyNoise,
    RangeResponseConfig,
    paths_to_frequency,
    validate_sample_times,
)

BANDWIDTH = 100e6
CARRIER = 28e9
PRF = 1000.0
GENERATION = "12345678-1234-5678-9234-567812345678"


class PulsePaths:
    """One independently solved pulse; path Doppler must never be consulted."""

    def __init__(self, gains, delays):
        self.gains = np.asarray(gains, dtype=np.complex128)
        self.delays = np.asarray(delays, dtype=np.float64)

    def cir(self, normalize_delays=False):
        assert normalize_delays is False
        return self.gains, self.delays

    @property
    def doppler(self):
        raise AssertionError("Sequential processing must use the acquired complex phase")


def processor_for(
    *,
    samples=16,
    bins=16,
    fft=16,
    frequency_window="boxcar",
    doppler_window="boxcar",
    clutter="none",
    noise=0.0,
    policy="always",
    **overrides,
):
    config = SensingConfig(
        enabled=True,
        processing_mode="sequential_stft",
        bandwidth=BANDWIDTH,
        prf_hz=PRF,
        fft_size_range=fft,
        fft_size_doppler=samples,
        window_range="boxcar",
        window_doppler=doppler_window,
        range_mode="monostatic",
        range_response=RangeResponseConfig(
            "cfr_ifft", bins, frequency_window, FrequencyNoise(noise, 123)
        ),
        coherent_clutter_mode=clutter,
        raw_payload_policy=policy,
        output_detections=False,
        output_range_profile=True,
        output_range_doppler=True,
        **overrides,
    )
    execution = resolve_sensing_execution_values(
        config, cir_time_steps=1, cir_sampling_frequency_hz=None, frame_dt_s=1 / PRF
    )
    return create_sensing_processor(config, CARRIER, execution)


def pulses_for(case, count):
    pulses = []
    for index in range(count):
        time = index / PRF
        delays = np.array([3 / BANDWIDTH])
        gains = np.array([0.7 + 0.3j])
        if case == "fractional":
            delays += 0.37 / BANDWIDTH
        elif case in {"approaching", "receding"}:
            sign = -1 if case == "approaching" else 1
            delays += sign * 125 * time / CARRIER
            gains *= np.exp(-2j * np.pi * CARRIER * (delays - 3 / BANDWIDTH))
        elif case == "oscillating":
            displacement = 0.15 * np.sin(2 * np.pi * 2 * time)
            delays += 2 * displacement / 299792458.0
            gains *= np.exp(-2j * np.pi * CARRIER * (delays - 3 / BANDWIDTH))
        elif case == "multipath":
            delays = np.array([3.37, 3.37, 8.21]) / BANDWIDTH
            gains = np.array([1 + 0.2j, -0.3 + 0.1j, 0.4 - 0.6j]) * np.exp(
                2j * np.pi * np.array([125, -125, 0]) * time
            )
        elif case == "changing_paths":
            count_here = index % 4
            delays = (3.13 + np.arange(count_here) * 2.12) / BANDWIDTH
            gains = (0.2 + np.arange(count_here) * 0.3j) * np.exp(0.37j * index)
        elif case == "empty":
            delays = np.empty(0)
            gains = np.empty(0, dtype=complex)
        pulses.append(PulsePaths(gains, delays))
    return pulses


def direct_fourier(pulses, *, bins, fft, frequency_window, doppler_window, clutter):
    """Scalar path sum and dense Fourier matrices, with no production FFT helpers."""
    q = np.arange(bins) - bins // 2
    frequencies = q * BANDWIDTH / bins
    observation = np.array(
        [
            [
                sum(a * np.exp(-2j * np.pi * f * tau) for a, tau in zip(pulse.gains, pulse.delays))
                for f in frequencies
            ]
            for pulse in pulses
        ],
        dtype=complex,
    )
    weights = np.ones(bins)
    if frequency_window == "hann":
        weights = np.sin(np.pi * np.arange(bins) / bins) ** 2
    synthesis = np.exp(2j * np.pi * q[:, None] * np.arange(fft)[None, :] / fft)
    # Project each acquired pulse independently. A batched BLAS product may
    # round identical rows differently, inventing variation that changes the
    # legacy operator's exact nonzero-variation decision below.
    delay = np.stack([(row * weights) @ synthesis / weights.sum() for row in observation])
    count = len(pulses)
    taper = np.ones(count)
    if doppler_window == "hann":
        taper = np.sin(np.pi * np.arange(count) / (count - 1)) ** 2
    cleaned = delay.copy()
    if clutter == "windowed_mean":
        centered = cleaned - cleaned[0]
        cleaned = centered - np.sum(centered * taper[:, None], axis=0) / taper.sum()
    elif clutter == "legacy" and np.any(np.diff(cleaned, axis=0)):
        cleaned -= cleaned.mean(axis=0)
    k = np.arange(count) - count // 2
    analysis = np.exp(-2j * np.pi * k[:, None] * np.arange(count)[None, :] / count)
    normalized_taper = taper / np.sqrt(np.sum(taper**2))
    return observation, delay, analysis @ (cleaned * normalized_taper[:, None])


def run_pulses(processor, pulses, *, start=2.0):
    results = []
    for index, pulse in enumerate(pulses):
        result = processor.process(None, index, pulse, source_time_s=start + index / PRF)
        assert result is not None
        results.append(result)
    return results


@pytest.mark.parametrize(
    "case",
    [
        "stationary",
        "fractional",
        "approaching",
        "receding",
        "oscillating",
        "multipath",
        "changing_paths",
        "empty",
    ],
)
@pytest.mark.parametrize(
    "frequency_window,doppler_window,fft",
    [
        ("boxcar", "boxcar", 16),
        ("hann", "hann", 16),
        ("hann", "boxcar", 31),
    ],
)
@pytest.mark.parametrize("clutter", ["none", "windowed_mean", "legacy"])
def test_complete_complex_sequential_output_matches_independent_fourier(
    case,
    frequency_window,
    doppler_window,
    fft,
    clutter,
):
    pulses = pulses_for(case, 17)
    processor = processor_for(
        frequency_window=frequency_window, doppler_window=doppler_window, fft=fft, clutter=clutter
    )
    if case == "empty":
        processor.config.output_detections = True
    results = run_pulses(processor, pulses)
    for end in (15, 16):
        observed, delay, rd = direct_fourier(
            pulses[end - 15 : end + 1],
            bins=16,
            fft=fft,
            frequency_window=frequency_window,
            doppler_window=doppler_window,
            clutter=clutter,
        )
        result = results[end]
        np.testing.assert_allclose(result["cfr_by_pair_rx_element"][0, 0], observed, atol=3e-13)
        np.testing.assert_allclose(result["slow_time_by_pair_rx_element"][0, 0], delay, atol=3e-13)
        np.testing.assert_allclose(result["range_doppler_by_pair"][0], rd, atol=2e-12)
        # Range profile measures the latest pulse, not average power over the STFT.
        np.testing.assert_allclose(
            result["range_profile_by_pair"][0], abs(delay[-1]) ** 2, atol=1e-13
        )
        if case == "empty":
            assert result["detections"] == []
    if case in {"approaching", "receding"} and clutter == "none":
        rd = results[-1]["range_doppler_by_pair"][0]
        peak_doppler = np.unravel_index(np.argmax(abs(rd)), rd.shape)[0]
        assert results[-1]["doppler_axis_hz"][peak_doppler] == (
            125 if case == "approaching" else -125
        )


def bind_results(processor, results, *, start=2.0):
    state = SensingPipelineState.from_config(
        SimpleNamespace(sensing={"enabled": True, "range_mode": "monostatic"}),
        frame_dt_s=1 / PRF,
        gt_dt_s=1 / PRF,
        first_step=0,
        source_generation_id=GENERATION,
        execution_config=processor.execution_config,
    )
    scheduler = FixedIntervalScheduler(first_step=0, last_step=len(results) - 1, interval_length=1)
    for index, sensing in enumerate(results):
        frame = {
            "frame_idx": index,
            "scene_time_s": start + index / PRF,
            "tx_list": [SimpleNamespace(name="tx")],
            "rx_list": [SimpleNamespace(name="rx")],
            "tx_positions_snapshot": np.zeros((1, 3)),
            "rx_positions_snapshot": np.zeros((1, 3)),
            # Actor state is supplied only after numerical processing, for the
            # pipeline's midpoint truth interpolation and evaluation metadata.
            "target_positions_snapshot": np.array([[10.0, 0.0, 0.0]]),
            "target_velocities_snapshot": np.zeros((1, 3)),
            "targets_metadata": [{"name": "fixture_target"}],
            "sensing": sensing,
        }
        state.process_frame(
            frame, step_idx=index, is_rt_step=True, schedule_event=scheduler.event_for_step(index)
        )
    return results


def test_warmup_63_64_65_exact_overlap_midpoint_and_bounded_histories():
    processor = processor_for(samples=64)
    results = run_pulses(processor, pulses_for("oscillating", 65))
    assert not any(result["rd_fresh"] for result in results[:63])
    assert results[63]["rd_fresh"] and results[64]["rd_fresh"]
    assert all(result["config_resolved"] == results[0]["config_resolved"] for result in results)
    assert len(processor._fast_history_raw) == 64
    assert len(processor._frequency_history) == 64
    assert len(processor._source_time_history) == 64
    bind_results(processor, results)
    for index in (0, 62):
        result = results[index]
        assert result["temporal"]["acquisition"]["observed_sample_count"] == index + 1
        assert result["temporal"]["acquisition"]["completeness"] == "incomplete_unprocessed"
        assert "result" not in result["temporal"]
        assert "range_doppler_by_pair" not in result
        assert result["cfr_by_pair_rx_element"].shape == (1, 1, 1, 16)
        with pytest.raises(ValueError, match="complete"):
            rebuild_frequency_observation(result)
    for index in (63, 64):
        temporal = results[index]["temporal"]
        assert temporal["acquisition"]["completeness"] == "complete"
        assert temporal["result"]["reference_time_s"] == pytest.approx(2 + (index - 31.5) / PRF)
        assert [
            sample["source"]["scene_frame_id"] for sample in temporal["acquisition"]["samples"]
        ] == list(range(index - 63, index + 1))
        assert all(
            sample["phase_coherent"] is False for sample in temporal["acquisition"]["samples"]
        )
    first, second = results[63], results[64]
    np.testing.assert_array_equal(
        first["cfr_by_pair_rx_element"][:, :, 1:], second["cfr_by_pair_rx_element"][:, :, :-1]
    )
    np.testing.assert_array_equal(
        first["slow_time_by_pair_rx_element"][:, :, 1:],
        second["slow_time_by_pair_rx_element"][:, :, :-1],
    )


@pytest.mark.parametrize("clutter", ["none", "legacy", "windowed_mean"])
def test_saved_sequential_replay_keeps_membership_phase_noise_and_last_sample_profile(
    tmp_path, clutter
):
    from shared.frames.frame_set_writer import FrameSetWriter
    from shared.sensing.loader import load_frames_sensing
    from tests.shared.test_sensing_postprocessor import _make_standard_frame

    processor = processor_for(
        samples=8, frequency_window="hann", doppler_window="hann", clutter=clutter, noise=0.002
    )
    results = bind_results(processor, run_pulses(processor, pulses_for("changing_paths", 9)))
    source = tmp_path / "observations"
    with FrameSetWriter.create_new(source, compression="lzf") as writer:
        for index in (6, 7, 8):
            writer.append(_make_standard_frame(frame_idx=index, sensing=results[index]))
        manifest = writer.finalize()
    # The existing writer rotates once when a one-pulse warmup observation
    # becomes a complete window; later overlapping windows share a chunk.
    assert len(manifest.chunks) == 2
    loaded = load_frames_sensing(source)
    assert len(loaded) == 3
    for frame, original in zip(loaded, results[6:]):
        saved = frame["_typed_sensing"]
        expected_temporal = deepcopy(original["temporal"])
        for sample in expected_temporal["acquisition"]["samples"]:
            if "source" in sample:
                sample["source"]["source_frame_set_id"] = manifest.frame_set_id
        assert saved["temporal"] == expected_temporal
        np.testing.assert_array_equal(
            saved["cfr_by_pair_rx_element"], original["cfr_by_pair_rx_element"]
        )
        if saved["temporal"]["acquisition"]["completeness"] != "complete":
            assert "result" not in saved["temporal"]
            with pytest.raises(ValueError, match="complete"):
                rebuild_frequency_observation(saved)
            continue
        observation_product = saved["products"]["cfr_by_pair_rx_element"]
        delay_product = saved["products"]["slow_time_by_pair_rx_element"]
        assert delay_product["lineage"]["source_product_ids"] == [observation_product["product_id"]]
        assert delay_product["processing"]["delay_projection"] == "cfr_ifft"
        assert delay_product["processing"]["frequency_window"] == "hann"
        rebuilt = SensingPostProcessor(
            PostProcessingConfig(skip_detection=True)
        ).process_from_frequency_observation(saved)
        np.testing.assert_allclose(
            rebuilt["rd_cube"], original["range_doppler_by_pair"][0], atol=2e-13, rtol=2e-13
        )
        np.testing.assert_allclose(
            rebuilt["cfr_rebuilt"]["range_profile"],
            original["range_profile_by_pair"][0],
            atol=2e-13,
        )
        np.testing.assert_array_equal(
            rebuilt["cfr_rebuilt"]["slow_time_axis_s"], original["slow_time_axis_s"]
        )


def test_noise_is_generated_once_per_pulse_and_reused_for_overlap():
    processor = processor_for(samples=8, noise=0.02)
    pulses = [PulsePaths([], []) for _ in range(12)]
    results = run_pulses(processor, pulses)
    repeated = run_pulses(processor_for(samples=8, noise=0.02), pulses)
    for index, result in enumerate(results):
        np.testing.assert_array_equal(
            result["cfr_by_pair_rx_element"], repeated[index]["cfr_by_pair_rx_element"]
        )
        expected = []
        for pulse_index in range(max(0, index - 7), index + 1) if index >= 7 else [index]:
            draws = np.random.default_rng(
                np.random.SeedSequence([123, pulse_index])
            ).standard_normal((16, 2))
            expected.append(np.sqrt(0.02 / 2) * (draws[:, 0] + 1j * draws[:, 1]))
        np.testing.assert_array_equal(result["cfr_by_pair_rx_element"][0, 0], expected)
    for first, second in zip(results[7:], results[8:]):
        np.testing.assert_array_equal(
            first["cfr_by_pair_rx_element"][:, :, 1:], second["cfr_by_pair_rx_element"][:, :, :-1]
        )


def test_single_projection_timestamp_does_not_permit_incomplete_doppler_window():
    config = RangeResponseConfig("cfr_ifft", 16)
    observed, _ = paths_to_frequency(
        np.array([[[1 + 2j]]]),
        np.array([3.37 / BANDWIDTH]),
        np.array([True]),
        np.array([7.2]),
        bandwidth=BANDWIDTH,
        config=config,
    )
    assert observed.shape == (1, 1, 16)
    with pytest.raises(ValueError):
        validate_sample_times(np.array([7.2]), 1)
    for times in (np.array([np.nan]), np.array([])):
        with pytest.raises(ValueError):
            paths_to_frequency(
                np.array([[[1 + 2j]]]),
                np.array([3.37 / BANDWIDTH]),
                np.array([True]),
                times,
                bandwidth=BANDWIDTH,
                config=config,
            )


@pytest.mark.parametrize("bad_time", [2.0, 1.999, 2.0015, 2.002, np.nan])
def test_sequential_rejects_nonuniform_nonfinite_or_nonincreasing_timestamps(bad_time):
    processor = processor_for(samples=4)
    pulse = PulsePaths([1 + 2j], [3.37 / BANDWIDTH])
    processor.process(None, 0, pulse, source_time_s=2.0)
    with pytest.raises(ValueError):
        processor.process(None, 1, pulse, source_time_s=bad_time)
    assert len(processor._source_time_history) == 1
    assert len(processor._frequency_history) == 1


@pytest.mark.parametrize("delay_bins", [16, 16.01])
def test_sequential_rejects_range_aliases_before_recording_pulse(delay_bins):
    processor = processor_for()
    with pytest.raises(ValueError, match="ambiguity period"):
        processor.process(
            None, 0, PulsePaths([1 + 0j], [delay_bins / BANDWIDTH]), source_time_s=0.0
        )
    assert not processor._source_time_history
    assert not processor._frequency_history


def test_causal_history_clutter_is_rejected_for_sequential_cfr():
    with pytest.raises((ValueError, ConfigurationError), match="[Cc]ausal|clutter"):
        processor_for(clutter_removal_enabled=True, clutter_removal_window=8)


def test_saved_sequential_replay_rejects_membership_corruption():
    processor = processor_for(samples=4)
    saved = bind_results(processor, run_pulses(processor, pulses_for("stationary", 4)))[-1]
    corrupted = deepcopy(saved)
    corrupted["temporal"]["acquisition"]["samples"][1]["sample_time_s"] += 0.0001
    with pytest.raises(ValueError, match="membership"):
        rebuild_frequency_observation(corrupted)


@pytest.mark.parametrize(
    "arguments,fft,frequency_window,doppler_window,clutter",
    [
        (["--range-ifft-size", "32", "--frequency-window", "hann"], 32, "hann", "boxcar", "none"),
        (["--rebuild-rd"], 16, "boxcar", "boxcar", "none"),
        (["--coherent-clutter-mode", "windowed_mean"], 16, "boxcar", "boxcar", "windowed_mean"),
        (["--window-doppler", "hann"], 16, "boxcar", "hann", "none"),
    ],
)
def test_cli_replay_keeps_warmup_and_exact_overlapping_windows(
    tmp_path, monkeypatch, arguments, fft, frequency_window, doppler_window, clutter
):
    from shared.frames.frame_set_writer import FrameSetWriter
    from shared.sensing.loader import load_frames_sensing
    from shared.sensing.postprocessor import main
    from tests.shared.test_sensing_postprocessor import _make_standard_frame

    processor = processor_for(samples=8, noise=0.002)
    results = bind_results(processor, run_pulses(processor, pulses_for("oscillating", 9)))
    source = tmp_path / "source"
    with FrameSetWriter.create_new(source, compression="lzf") as writer:
        for index, sensing in enumerate(results):
            frame = _make_standard_frame(frame_idx=index, sensing=sensing)
            writer.append(replace(frame, timestamp_s=2 + index / PRF))
        writer.finalize()
    source_loaded = load_frames_sensing(source)
    destination = tmp_path / "replayed"
    monkeypatch.setattr(
        "sys.argv",
        [
            "shared.sensing.postprocessor",
            str(source),
            "--output-dir",
            str(destination),
            *arguments,
            "--skip-detection",
        ],
    )
    main()
    replayed = load_frames_sensing(destination)
    assert len(replayed) == 9
    for index, frame in enumerate(replayed):
        saved = frame["_typed_sensing"]
        original = source_loaded[index]["_typed_sensing"]
        if index < 7 or fft == 32:
            np.testing.assert_array_equal(
                saved["cfr_by_pair_rx_element"], original["cfr_by_pair_rx_element"]
            )
        assert saved["temporal"]["acquisition"] == original["temporal"]["acquisition"]
        if index < 7:
            assert "result" not in saved["temporal"]
            assert "range_doppler_complex_by_pair" not in saved
            assert "range_doppler_by_pair" not in saved
        else:
            assert saved["temporal"]["result"]["reference_time_s"] == pytest.approx(
                2 + (index - 3.5) / PRF
            )
            assert saved["range_doppler_power_by_pair"].shape == (1, 8, fft)
            expected = rebuild_frequency_observation(
                original,
                fft_size=fft,
                response=replace(
                    processor.config.range_response, frequency_window=frequency_window
                ),
                window_doppler=doppler_window,
                coherent_clutter_mode=clutter,
            )
            np.testing.assert_allclose(
                saved["range_doppler_power_by_pair"][0], abs(expected["rd_cube"]) ** 2, atol=2e-13
            )
            if fft == 32:
                np.testing.assert_array_equal(
                    saved["range_doppler_complex_by_pair"][0], expected["rd_cube"]
                )
                np.testing.assert_allclose(
                    saved["range_profile_by_pair"][0], expected["range_profile"], atol=2e-13
                )
            if doppler_window == "hann" or clutter == "windowed_mean":
                assert not np.allclose(
                    saved["range_doppler_power_by_pair"][0],
                    abs(original["range_doppler_by_pair"][0]) ** 2,
                )
