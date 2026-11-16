"""Analytical and native-boundary regressions for sequential delay-bin sensing."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from generator.core.configuration import SensingConfig
from generator.core.pipeline.schedule import FixedIntervalScheduler
from generator.core.pipeline.sensing import SensingPipelineState
from generator.core.sensing.acquisition import resolve_sensing_execution_values
from generator.core.sensing.processor import create_sensing_processor
from shared.extensions.sensing import assert_sensing_contract


class _Pulse:
    def __init__(self, amplitude=1.0 + 0j, *, fail=False):
        self.amplitude = amplitude
        self.fail = fail

    def cir(self, normalize_delays=False):
        if self.fail:
            raise RuntimeError("controlled missing CIR pulse")
        return np.array([self.amplitude]), np.array([4 / 1e8])


def _processor(**overrides):
    options = dict(
        enabled=True,
        processing_mode="sequential_stft",
        range_mode="monostatic",
        bandwidth=1e8,
        prf_hz=1000.0,
        fft_size_range=16,
        fft_size_doppler=4,
        window_range="boxcar",
        window_doppler="boxcar",
        clutter_removal_window=2,
        output_range_profile=True,
        output_range_doppler=True,
        output_detections=False,
        raw_payload_policy="always",
        cfar_guard_cells=0,
        cfar_ref_cells=2,
        cfar_threshold_scale=1.5,
        cfar_type="CA",
    )
    options.update(overrides)
    config = SensingConfig(**options)
    execution = resolve_sensing_execution_values(
        config, cir_time_steps=1, cir_sampling_frequency_hz=None, frame_dt_s=0.001
    )
    return create_sensing_processor(config, 28e9, execution)


@pytest.mark.parametrize("ramp", [False, True])
@pytest.mark.parametrize("doppler_window", ["boxcar", "hann"])
def test_history_cleaned_constant_input_is_transformed_even_without_mean_removal(
    ramp, doppler_window
):
    processor = _processor(clutter_removal_enabled=True, window_doppler=doppler_window)
    for index in range(8):
        result = processor.process(
            None, index, _Pulse(index + 1 if ramp else 1.0), source_time_s=index / 1000
        )
    # A static sequence is zero after causal subtraction; a unit ramp minus
    # its previous two samples' mean is exactly 1.5 after warm-up.
    cleaned = np.full(4, 1.5 if ramp else 0.0)
    taper = np.hanning(4) if doppler_window == "hann" else np.ones(4)
    expected = np.fft.fftshift(np.fft.fft(cleaned * taper)) / np.linalg.norm(taper)
    np.testing.assert_allclose(result["range_doppler_by_pair"][0, :, 4], expected, atol=1e-14)
    assert np.any(result["range_doppler_raw_by_pair"])
    spec = next(s for s in result["_product_specs"] if s["data_key"] == "range_doppler_by_pair")
    assert spec["processing"]["slow_time_mean_removal_applied"] is False
    assert result["clutter_metadata"]["range_doppler"]["steady_state"]


@pytest.mark.parametrize("policy", ["none", "always"])
@pytest.mark.parametrize("window", ["boxcar", "hann"])
@pytest.mark.parametrize("clutter", [False, True])
def test_noisy_profile_and_cfar_share_the_acquired_sample_before_history_clutter(
    policy, window, clutter
):
    processor = _processor(
        noise_enabled=True,
        noise_snr_db=0.0,
        rng_seed=7,
        raw_payload_policy=policy,
        window_range=window,
        clutter_removal_enabled=clutter,
    )
    rng = np.random.default_rng(7)
    taper = np.hanning(16) if window == "hann" else np.ones(16)
    taper *= np.sqrt(16 / np.sum(taper**2))
    powers = []
    for index in range(4):
        # Unit signal power at 0 dB SNR gives variance 1/2 per quadrature.
        acquired = (rng.standard_normal(16) + 1j * rng.standard_normal(16)) / np.sqrt(2)
        acquired[4] += 1
        raw_power = abs(acquired * taper) ** 2
        expected = raw_power
        if clutter and powers:
            expected = np.maximum(raw_power - np.mean(powers[-2:], axis=0), 0)
        result = processor.process(None, index, _Pulse(), source_time_s=index / 1000)
        np.testing.assert_allclose(result["range_profile_by_pair"][0], expected, atol=1e-14)
        # Explicit CA reference ring: two bins on each side, no guard, full windows.
        mask = np.zeros(16, dtype=np.uint8)
        for cut in range(2, 14):
            training = [expected[cell] for cell in (cut - 2, cut - 1, cut + 1, cut + 2)]
            mask[cut] = expected[cut] > 1.5 * np.mean(training)
        np.testing.assert_array_equal(result["range_profile_cfar_mask"], mask)
        if policy == "always":
            np.testing.assert_allclose(result["cir_by_pair_rx_element"][0, 0], acquired, atol=1e-14)
            np.testing.assert_allclose(
                result["range_profile_raw_by_pair"][0], raw_power, atol=1e-14
            )
        else:
            assert "cir_by_pair_rx_element" not in result
        powers.append(raw_power)


@pytest.mark.parametrize("profile_output", [False, True])
def test_noisy_profile_cfar_remains_available_without_rd_or_retained_cir(profile_output):
    processor = _processor(
        noise_enabled=True,
        noise_snr_db=0.0,
        rng_seed=7,
        output_range_profile=profile_output,
        output_range_doppler=False,
        output_detections=True,
        raw_payload_policy="none",
    )
    result = processor.process(None, 0, _Pulse(), source_time_s=0.0)
    assert "range_profile_cfar_mask" in result
    assert "cir" not in result and "range_doppler_by_pair" not in result
    assert ("range_profile_by_pair" in result) is profile_output
    reference = _processor(noise_enabled=True, noise_snr_db=0.0, rng_seed=7)
    retained = reference.process(None, 0, _Pulse(), source_time_s=0.0)
    np.testing.assert_array_equal(
        result["range_profile_cfar_mask"], retained["range_profile_cfar_mask"]
    )


@pytest.mark.parametrize("timestamp", [0.0, -0.001, 0.0015, 0.002, np.nan])
def test_invalid_cadence_rejects_before_noise_or_history_mutation(timestamp):
    processor = _processor(noise_enabled=True, noise_snr_db=0.0, rng_seed=7)
    processor.process(None, 0, _Pulse(), source_time_s=0.0)
    rng_state = deepcopy(processor._noise_rng.bit_generator.state)
    original = np.array(processor._fast_history_raw, copy=True)
    with pytest.raises(ValueError):
        processor.process(None, 1, _Pulse(), source_time_s=timestamp)
    assert list(processor._source_time_history) == [0.0]
    assert processor._noise_rng.bit_generator.state == rng_state
    np.testing.assert_array_equal(processor._fast_history_raw, original)


@pytest.mark.parametrize("failed_step", [2, 5])
def test_missing_cir_cannot_publish_compressed_time_through_service_and_native_pipeline(
    monkeypatch, failed_step
):
    from generator.core.services import raytracing_service as module

    processor = _processor()
    simulation_config = SimpleNamespace(num_steps=9, duration=0.008, sensing=processor.config)
    monkeypatch.setattr(module, "PathSolver", lambda: object())
    monkeypatch.setattr(module, "freeze_frame_paths", lambda frame: None)

    def ray_frame(_objects, index, **_kwargs):
        return {
            "frame_idx": index,
            "paths": _Pulse(np.exp(2j * np.pi * 250 * index / 1000), fail=index == failed_step),
            "tx_list": [SimpleNamespace(name="tx")],
            "rx_list": [SimpleNamespace(name="rx")],
            "tx_positions_snapshot": np.zeros((1, 3)),
            "rx_positions_snapshot": np.zeros((1, 3)),
            "target_positions_snapshot": np.array([[10.0, 0.0, 0.0]]),
            "target_velocities_snapshot": np.zeros((1, 3)),
            "targets_metadata": [{"name": "target"}],
        }

    monkeypatch.setattr(module, "compute_ray_tracing_step", ray_frame)
    service = module.RayTracingService(simulation_config)
    service.simulation_objects = SimpleNamespace()
    service.sensing_processor = processor
    monkeypatch.setattr(service, "_compute_cir_stack", lambda paths: None)
    state = SensingPipelineState.from_config(
        SimpleNamespace(sensing={"enabled": True, "range_mode": "monostatic"}),
        frame_dt_s=0.001,
        gt_dt_s=0.001,
        first_step=0,
        source_generation_id="12345678-1234-5678-9234-567812345678",
        execution_config=processor.execution_config,
    )
    scheduler = FixedIntervalScheduler(first_step=0, last_step=8, interval_length=1)
    complete_steps = []
    for index in range(9):
        frame = service.compute_step(index)
        if index > failed_step:
            assert frame is None
            continue
        assert frame is not None
        state.process_frame(
            frame, step_idx=index, is_rt_step=True, schedule_event=scheduler.event_for_step(index)
        )
        sensing = frame.get("sensing")
        if sensing is not None:
            assert_sensing_contract(frame, require_schema_version=True)
            if sensing["temporal"]["acquisition"]["completeness"] == "complete":
                complete_steps.append(index)
                np.testing.assert_allclose(np.diff(sensing["slow_time_axis_s"]), 0.001)
    assert complete_steps == list(range(3, failed_step))
    assert processor._source_time_history[-1] == (failed_step - 1) / 1000
    assert len(state.source_history) == min(failed_step, 4)
