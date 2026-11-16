"""Independent finite-record clutter checks on both native range projections."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from generator.core.configuration import AntennaConfig, SensingConfig
from generator.core.exceptions import ConfigurationError
from generator.core.sensing.acquisition import resolve_sensing_execution_values
from generator.core.sensing.processor import create_sensing_processor
from shared.sensing.postprocessor import PostProcessingConfig, SensingPostProcessor
from shared.sensing.range_response import RangeResponseConfig
from shared.sensing.signal_processing import remove_window_weighted_mean


@pytest.mark.parametrize("model", ["point_bin_floor", "cfr_ifft"])
@pytest.mark.parametrize("mode", ["none", "legacy", "windowed_mean"])
@pytest.mark.parametrize("window", ["boxcar", "hann", "hamming"])
@pytest.mark.parametrize("rows,columns", [(1, 1), (1, 4), (2, 2)])
def test_native_clutter_matches_independent_dft(model, mode, window, rows, columns):
    n = 64
    config = SensingConfig(
        enabled=True,
        processing_mode="coherent_cpi",
        coherent_clutter_mode=mode,
        bandwidth=500e6,
        prf_hz=1000,
        fft_size_range=128,
        fft_size_doppler=n,
        window_range="boxcar",
        window_doppler=window,
        range_mode="monostatic",
        raw_payload_policy="always",
        output_detections=False,
        output_range_profile=True,
        range_response=(
            RangeResponseConfig("cfr_ifft", 128) if model == "cfr_ifft" else RangeResponseConfig()
        ),
    )
    elements = rows * columns
    execution = resolve_sensing_execution_values(
        config,
        cir_time_steps=n,
        cir_sampling_frequency_hz=1000,
        frame_dt_s=0.001,
        rx_elements=elements,
    )
    processor = create_sensing_processor(config, 28e9, execution)
    # Three coherent returns: static, off-grid fast motion, and near-DC motion.
    tones = np.array([1 + 2j, 0.3 - 0.7j, 0.4 + 0.2j])[:, None] * np.exp(
        2j * np.pi * np.array([0, 336.23260796, 1.5625])[:, None] * np.arange(n) / 1000
    )
    amplitudes = np.exp(0.4j * np.arange(elements))[:, None, None] * tones[None]
    sim = SimpleNamespace(
        simulation_config=SimpleNamespace(rx_antenna=AntennaConfig(num_rows=rows, num_cols=columns))
    )
    result = processor.process(
        sim,
        63,
        None,
        cir_stack=(amplitudes[None, :, None, None], np.array([[[30.2e-9, 46.3e-9, 64.1e-9]]])),
        source_time_s=0,
    )
    samples = result["slow_time_by_pair_rx_element"][0]
    w = {"boxcar": np.ones, "hann": np.hanning, "hamming": np.hamming}[window](n)
    clean = samples.copy()
    if mode != "none":
        weights = w if mode == "windowed_mean" else np.ones(n)
        clean -= np.sum(samples * weights[None, :, None], axis=1, keepdims=True) / weights.sum()
    ell = np.arange(-n // 2, n // 2)
    dft = np.exp(-2j * np.pi * ell[:, None] * np.arange(n) / n) / np.sqrt(n)
    gain = np.sqrt(n / np.sum(w * w))
    expected = dft @ (clean.sum(axis=0) / np.sqrt(elements) * w[:, None] * gain)
    raw = dft @ (samples.sum(axis=0) / np.sqrt(elements) * w[:, None] * gain)
    scale = np.max(abs(raw))
    np.testing.assert_allclose(
        result["range_doppler_by_pair"][0] / scale, expected / scale, atol=1e-14, rtol=2e-13
    )
    np.testing.assert_allclose(
        result["range_doppler_raw_by_pair"][0] / scale, raw / scale, atol=1e-14, rtol=2e-13
    )
    replay = SensingPostProcessor(
        PostProcessingConfig(
            skip_detection=True,
            window_range="boxcar",
            window_doppler=window,
            coherent_clutter_mode=mode,
        )
    ).process_from_slow_time_matrix(
        samples.sum(axis=0) / np.sqrt(elements),
        {"config": {**asdict(config), "carrier_freq_hz": 28e9}},
        range_axis_m=result["range_axis_m"],
    )
    np.testing.assert_allclose(replay["rd_cube"] / scale, expected / scale, atol=1e-14, rtol=2e-13)
    np.testing.assert_allclose(
        result["range_profile_by_pair"][0],
        abs(samples[:, -1].sum(axis=0) / np.sqrt(elements)) ** 2,
        rtol=2e-14,
        atol=1e-25,
    )
    assert result["slow_time_mean_removal_applied"] is (mode != "none")
    if mode == "windowed_mean":
        assert np.max(abs(result["range_doppler_by_pair"][0, n // 2])) / scale < 2e-15
    if mode == "none":
        np.testing.assert_array_equal(
            result["range_doppler_by_pair"], result["range_doppler_raw_by_pair"]
        )
        assert result["slow_time_mean_removal_policy"] == "disabled"


@pytest.mark.parametrize("window", [np.ones, np.hanning, np.hamming, np.blackman])
def test_weighted_projection_constant_and_low_velocity_tradeoff(window):
    w = window(64)
    static = np.full((3, 64, 5), 1000 + 2000j)
    np.testing.assert_array_equal(
        remove_window_weighted_mean(static, w, axis=1), np.zeros_like(static)
    )
    losses = []
    for bins in (0.1, 0.25, 0.5, 1, 2, 21.5188869):
        x = np.exp(2j * np.pi * bins * np.arange(64) / 64)
        y = remove_window_weighted_mean(x, w, axis=0)
        np.testing.assert_allclose(np.sum(w * y), 0, atol=4e-14)
        np.testing.assert_allclose(
            remove_window_weighted_mean(x + 1000 + 2000j, w, axis=0), y, atol=4e-13
        )
        losses.append(10 * np.log10(np.sum(abs(w * y) ** 2) / np.sum(abs(w * x) ** 2)))
    assert losses[0] < -14
    assert abs(losses[-1]) < 0.02


def test_omitted_mode_retains_legacy_and_invalid_modes_fail():
    assert SensingConfig().coherent_clutter_mode == "legacy"
    with pytest.raises(ConfigurationError, match="coherent_clutter_mode"):
        SensingConfig(coherent_clutter_mode="mean")


@pytest.mark.parametrize("model", ["point_bin_floor", "cfr_ifft"])
@pytest.mark.parametrize("elements", [1, 4])
@pytest.mark.parametrize("window", ["boxcar", "hann", "hamming"])
def test_explicit_legacy_is_numerically_identical_to_omission(model, elements, window):
    """Declaring the compatibility recipe preserves arrays and detections across CPIs."""
    options = dict(
        enabled=True,
        processing_mode="coherent_cpi",
        bandwidth=500e6,
        prf_hz=1000,
        fft_size_range=128,
        fft_size_doppler=64,
        window_range="boxcar",
        window_doppler=window,
        range_mode="monostatic",
        raw_payload_policy="always",
        output_detections=True,
        output_range_profile=True,
        range_response=(
            RangeResponseConfig(model, 128) if model == "cfr_ifft" else RangeResponseConfig()
        ),
    )
    implicit = SensingConfig(**options)
    explicit = SensingConfig(**options, coherent_clutter_mode="legacy")
    execution = resolve_sensing_execution_values(
        implicit,
        cir_time_steps=64,
        cir_sampling_frequency_hz=1000,
        frame_dt_s=0.001,
        rx_elements=elements,
    )
    processors = [create_sensing_processor(c, 28e9, execution) for c in (implicit, explicit)]
    sim = SimpleNamespace(
        simulation_config=SimpleNamespace(rx_antenna=AntennaConfig(num_rows=1, num_cols=elements))
    )
    for cpi in range(3):
        t = (np.arange(64) + 64 * cpi) / 1000
        tones = np.array([1 + 2j, 0.3 - 0.7j])[:, None] * np.exp(
            2j * np.pi * np.array([0, 336.2])[:, None] * t
        )
        amplitudes = np.exp(0.4j * np.arange(elements))[:, None, None] * tones[None]
        results = [
            p.process(
                sim,
                63 + 64 * cpi,
                None,
                cir_stack=(amplitudes[None, :, None, None], np.array([[[30.2e-9, 46.3e-9]]])),
                source_time_s=0.064 * cpi,
            )
            for p in processors
        ]
        for key in (
            "range_doppler_by_pair",
            "range_doppler_raw_by_pair",
            "range_profile_by_pair",
            "range_axis_m",
            "doppler_axis_hz",
        ):
            np.testing.assert_array_equal(results[0][key], results[1][key])
        assert len(results[0]["detections"]) == len(results[1]["detections"])
        for a, b in zip(results[0]["detections"], results[1]["detections"]):
            assert a.keys() == b.keys()
            np.testing.assert_equal(a, b)
