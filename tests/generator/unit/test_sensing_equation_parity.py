"""Generator/shared parity against independent Phase 1B equation references."""

from __future__ import annotations

import numpy as np

from generator.core.configuration import SensingConfig
from generator.core.sensing.acquisition import resolve_sensing_execution_values
from generator.core.sensing.processor import create_sensing_processor
from shared.sensing.postprocessor import PostProcessingConfig, SensingPostProcessor
from shared.sensing.signal_processing import (
    DELAY_PROJECTION_POINT_BIN_FLOOR,
    DOPPLER_FFT_NORMALIZATION_UNITARY,
    RANGE_PROCESSING_DELAY_DOMAIN_CIR_BINS,
    WINDOW_NORMALIZATION_ENERGY,
    compute_range_doppler,
)
from tests.shared.sensing_equation_reference import (
    centered_unitary_dft,
    complex_tone,
    energy_correction,
    point_bin_floor_projection,
    reference_window,
)


class _SequentialPaths:
    def __init__(self, amplitudes: np.ndarray, delay_s: float) -> None:
        self._amplitudes = amplitudes
        self._delay_s = delay_s
        self._index = 0

    def cir(self, normalize_delays: bool = False) -> tuple[np.ndarray, np.ndarray]:
        del normalize_delays
        amplitude = self._amplitudes[self._index]
        self._index += 1
        return (
            np.array([amplitude], dtype=np.complex128),
            np.array([self._delay_s], dtype=np.float64),
        )


def _config(*, processing_mode: str) -> SensingConfig:
    return SensingConfig(
        enabled=True,
        bandwidth=1.0e9,
        prf_hz=800.0,
        fft_size_range=16,
        fft_size_doppler=8,
        window_range="blackman-harris",
        window_doppler="hamming",
        output_range_profile=False,
        output_range_doppler=True,
        output_detections=False,
        raw_payload_policy="diagnostic",
        processing_mode=processing_mode,
    )


def _processor(config: SensingConfig, *, cir_steps: int):
    resolved = resolve_sensing_execution_values(
        config,
        cir_time_steps=cir_steps,
        cir_sampling_frequency_hz=(config.prf_hz if cir_steps > 1 else None),
        frame_dt_s=1.0 / config.prf_hz,
    )
    return create_sensing_processor(config, 28.0e9, resolved)


def _independent_expected(
    amplitudes: np.ndarray,
    *,
    delay_s: float,
    config: SensingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    delay_rows = np.stack(
        [
            point_bin_floor_projection(
                [amplitude],
                [delay_s],
                sampling_frequency_hz=config.bandwidth,
                bin_count=config.fft_size_range,
            )
            for amplitude in amplitudes
        ]
    )
    delay_taper = reference_window(config.window_range, config.fft_size_range)
    delay_data = delay_rows * delay_taper[np.newaxis, :] * energy_correction(delay_taper)
    delay_data -= delay_data.mean(axis=0, keepdims=True)
    doppler_taper = reference_window(config.window_doppler, config.fft_size_doppler)
    fft_input = delay_data * doppler_taper[:, np.newaxis] * energy_correction(doppler_taper)
    expected = np.stack(
        [
            centered_unitary_dft(fft_input[:, delay_bin], prf_hz=config.prf_hz)
            for delay_bin in range(config.fft_size_range)
        ],
        axis=1,
    )
    return delay_rows, expected


def _assert_semantics(result: dict[str, object]) -> None:
    assert result["delay_projection"] == DELAY_PROJECTION_POINT_BIN_FLOOR
    assert result["range_processing"] == RANGE_PROCESSING_DELAY_DOMAIN_CIR_BINS
    assert result["range_fft_applied"] is False
    assert result["fft_axis"] == "slow_time"
    assert result["fft_normalization"] == DOPPLER_FFT_NORMALIZATION_UNITARY
    assert result["window_normalization"] == WINDOW_NORMALIZATION_ENERGY
    assert result["rd_model"] == "point_binned_cir_doppler_fft"


def _assert_generator_and_shared_match_reference(
    result: dict[str, object],
    *,
    delay_rows: np.ndarray,
    expected: np.ndarray,
    config: SensingConfig,
) -> None:
    generated = np.asarray(result["range_doppler_map"])
    np.testing.assert_allclose(generated, expected, rtol=8.0e-13, atol=8.0e-13)

    shared_result = compute_range_doppler(
        fast_time_sequence=delay_rows,
        sampling_frequency=config.bandwidth,
        prf_hz=config.prf_hz,
        fft_size_range=config.fft_size_range,
        fft_size_doppler=config.fft_size_doppler,
        window_range=config.window_range,
        window_doppler=config.window_doppler,
    )
    assert shared_result is not None and shared_result.cube is not None
    np.testing.assert_allclose(shared_result.cube, expected, rtol=8.0e-13, atol=8.0e-13)
    np.testing.assert_allclose(generated, shared_result.cube, rtol=8.0e-13, atol=8.0e-13)

    offline = SensingPostProcessor(
        PostProcessingConfig(
            window_range=config.window_range,
            window_doppler=config.window_doppler,
            mti_enabled=True,
            skip_detection=True,
        )
    ).process_from_slow_time_matrix(
        delay_rows,
        {
            "config": {
                "fft_size_range": config.fft_size_range,
                "fft_size_doppler": config.fft_size_doppler,
                "sampling_frequency": config.bandwidth,
                "prf_hz": config.prf_hz,
                "range_mode": config.range_mode,
                "carrier_freq_hz": 28.0e9,
            }
        },
    )
    np.testing.assert_allclose(offline["rd_cube"], expected, rtol=8.0e-13, atol=8.0e-13)
    np.testing.assert_allclose(generated, offline["rd_cube"], rtol=8.0e-13, atol=8.0e-13)
    _assert_semantics(result)


def test_coherent_generator_matches_independent_unitary_reference() -> None:
    config = _config(processing_mode="coherent_cpi")
    amplitudes = complex_tone(200.0, prf_hz=config.prf_hz, sample_count=8)
    delay_s = 5.25 / config.bandwidth
    delay_rows, expected = _independent_expected(
        amplitudes,
        delay_s=delay_s,
        config=config,
    )
    processor = _processor(config, cir_steps=8)

    result = processor.process(
        None,
        frame_idx=7,
        paths=None,
        cir_stack=(amplitudes[np.newaxis, :], np.array([delay_s])),
        source_time_s=0.01,
    )

    assert result is not None
    _assert_generator_and_shared_match_reference(
        result,
        delay_rows=delay_rows,
        expected=expected,
        config=config,
    )


def test_sequential_generator_matches_independent_unitary_reference() -> None:
    config = _config(processing_mode="sequential_stft")
    amplitudes = complex_tone(-200.0, prf_hz=config.prf_hz, sample_count=8)
    delay_s = 5.75 / config.bandwidth
    delay_rows, expected = _independent_expected(
        amplitudes,
        delay_s=delay_s,
        config=config,
    )
    paths = _SequentialPaths(amplitudes, delay_s)
    processor = _processor(config, cir_steps=1)

    result = None
    for frame_index in range(8):
        result = processor.process(
            None,
            frame_idx=frame_index,
            paths=paths,
            source_time_s=frame_index / config.prf_hz,
        )

    assert result is not None
    _assert_generator_and_shared_match_reference(
        result,
        delay_rows=delay_rows,
        expected=expected,
        config=config,
    )
