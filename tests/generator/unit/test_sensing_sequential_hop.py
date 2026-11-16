"""Output cadence changes complete windows, never the acquired pulse stream."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from generator.core.configuration import SensingConfig
from generator.core.configuration.loader import load_simulation_config
from generator.core.exceptions import ConfigurationError
from generator.core.pipeline.schedule import FixedIntervalScheduler
from generator.core.pipeline.sensing import SensingPipelineState
from generator.core.sensing.acquisition import resolve_sensing_execution_values
from generator.core.sensing.processor import create_sensing_processor
from shared.scenarios.model import SensingModel
from shared.sensing.range_response import FrequencyNoise, RangeResponseConfig
from tests.shared.test_sensing_sequential_cfr import GENERATION, PulsePaths

PRF = 4000.0


def _processor(hop=None, *, samples=8, **overrides):
    values = dict(
        enabled=True,
        processing_mode="sequential_stft",
        cpi_hop_samples=hop,
        bandwidth=100e6,
        prf_hz=PRF,
        fft_size_range=4,
        fft_size_doppler=samples,
        window_range="boxcar",
        window_doppler="hann",
        range_mode="monostatic",
        range_response=RangeResponseConfig("cfr_ifft", 4, "boxcar", FrequencyNoise(0.001, 17)),
        coherent_clutter_mode="none",
        output_detections=False,
        output_range_profile=True,
        output_range_doppler=True,
        recording_profile="reprocessing",
    )
    values.update(overrides)
    config = SensingConfig(**values)
    execution = resolve_sensing_execution_values(
        config, cir_time_steps=1, cir_sampling_frequency_hz=None, frame_dt_s=1 / PRF
    )
    return create_sensing_processor(config, 28e9, execution)


def _run(processor, count):
    state = SensingPipelineState.from_config(
        SimpleNamespace(sensing={"enabled": True, "range_mode": "monostatic"}),
        frame_dt_s=1 / PRF,
        gt_dt_s=1 / PRF,
        first_step=0,
        source_generation_id=GENERATION,
        execution_config=processor.execution_config,
    )
    scheduler = FixedIntervalScheduler(first_step=0, last_step=count - 1, interval_length=1)
    results = []
    for index in range(count):
        sensing = processor.process(
            None,
            index,
            PulsePaths([np.exp(2j * np.pi * 375 * index / PRF)], [1.37 / 100e6]),
            source_time_s=index / PRF,
        )
        frame = dict(
            frame_idx=index,
            scene_time_s=index / PRF,
            tx_list=[SimpleNamespace(name="tx")],
            rx_list=[SimpleNamespace(name="rx")],
            tx_positions_snapshot=np.zeros((1, 3)),
            rx_positions_snapshot=np.zeros((1, 3)),
            target_positions_snapshot=np.array([[10.0, 0.0, 0.0]]),
            target_velocities_snapshot=np.zeros((1, 3)),
            targets_metadata=[{"name": "fixture_target"}],
            sensing=sensing,
        )
        state.process_frame(
            frame, step_idx=index, is_rt_step=True, schedule_event=scheduler.event_for_step(index)
        )
        results.append(sensing)
    return results


def test_grouped_hop_reaches_runtime_without_changing_pulse_clock():
    authored = SensingModel.model_validate(
        {
            "enabled": True,
            "acquisition": {"mode": "sequential_stft", "sample_rate_hz": PRF},
            "processing": {"doppler": {"fft_size": 256, "hop_samples": 64}},
        }
    )
    config = load_simulation_config(SimpleNamespace(sensing=authored.model_dump(exclude_none=True)))
    execution = resolve_sensing_execution_values(
        config.sensing, cir_time_steps=1, cir_sampling_frequency_hz=None, frame_dt_s=1 / PRF
    )
    assert config.sensing.cpi_hop_samples == 64
    assert execution.acquisition.cpi_hop_samples == execution.processing.result_stride_samples == 64
    assert execution.acquisition.overlap_samples == 192
    assert execution.acquisition.rt_solve_stride_scene_steps == 1
    assert execution.acquisition.sample_rate_hz == PRF


@pytest.mark.parametrize("hop", [True, False, 0, -1, 1.5, "64", 257])
def test_invalid_hop_is_rejected_without_coercion(hop):
    with pytest.raises((ConfigurationError, ValidationError), match="hop"):
        values = SensingModel.model_validate(
            {
                "enabled": True,
                "processing_mode": "sequential_stft",
                "fft_size_doppler": 256,
                "cpi_hop_samples": hop,
            }
        ).model_dump(exclude_none=True)
        load_simulation_config(SimpleNamespace(sensing=values))
    with pytest.raises(ConfigurationError, match="hop"):
        SensingConfig(cpi_hop_samples=hop, fft_size_doppler=256, processing_mode="sequential_stft")


@pytest.mark.parametrize("mode", ["coherent_cpi", "analytic_path_doppler_oracle", "auto"])
def test_hop_cannot_override_other_modes(mode):
    with pytest.raises(ConfigurationError, match="hop.*sequential"):
        config = SensingConfig(processing_mode=mode, cpi_hop_samples=1, fft_size_doppler=8)
        resolve_sensing_execution_values(
            config, cir_time_steps=8, cir_sampling_frequency_hz=config.prf_hz, frame_dt_s=1 / PRF
        )


def test_omitted_hop_preserves_explicit_one_pulse_products_and_identities():
    default, explicit = _run(_processor(), 20), _run(_processor(1), 20)
    for before, after in zip(default, explicit, strict=True):
        assert before["config_resolved"] == after["config_resolved"]
        assert before["temporal"] == after["temporal"]
        assert before["products"] == after["products"]
        for key, value in before.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, after[key])


@pytest.mark.parametrize("hop", [64, 128, 256])
def test_native_windows_preserve_samples_and_bind_only_on_cadence(hop):
    processor = _processor(hop, samples=256)
    results = _run(processor, 512)
    expected = list(range(255, 512, hop))
    assert [i for i, row in enumerate(results) if row["rd_fresh"]] == expected
    assert [i for i, row in enumerate(results) if "result" in row["temporal"]] == expected
    for index, row in enumerate(results):
        temporal = row["temporal"]
        acquisition = temporal["acquisition"]
        if index < 255:
            assert acquisition["completeness"] == "incomplete_unprocessed"
        else:
            assert acquisition["completeness"] == "complete"
            assert [s["source"]["scene_frame_id"] for s in acquisition["samples"]] == list(
                range(index - 255, index + 1)
            )
        if index not in expected:
            assert "result" not in temporal and "display_binding" not in temporal
            assert "truth" not in temporal and "gt" not in row
            assert "range_doppler_power_by_pair" not in row and "detections" not in row
        else:
            assert temporal["result"]["reference_time_s"] == pytest.approx((index - 127.5) / PRF)
            assert temporal["display_binding"]["scene_frame_id"] == index - 128
    assert processor._source_sample_count == 512
    assert len(processor._fast_history_raw) == len(processor._source_time_history) == 256
    plan = processor.execution_config.acquisition
    full_ends = [count - 1 for count in range(1, 8193) if plan.window_due(count)]
    assert len(full_ends) == 1 + (8192 - 256) // hop
    assert full_ends == list(range(255, 8192, hop))


@pytest.mark.parametrize("cfr", [False, True])
def test_hop_changes_no_observation_noise_or_clutter_values(cfr):
    settings = (
        {"coherent_clutter_mode": "windowed_mean"}
        if cfr
        else {
            "range_response": RangeResponseConfig(),
            "coherent_clutter_mode": "legacy",
            "clutter_removal_enabled": True,
            "clutter_removal_window": 3,
            "noise_enabled": True,
            "noise_snr_db": 12,
            "rng_seed": 17,
        }
    )
    dense = _run(_processor(1, **settings), 24)
    sparse = _run(_processor(4, **settings), 24)
    for index, row in enumerate(sparse):
        # CFR reprocessing keeps the frequency observation and derives delay
        # samples from it; it does not retain a redundant complex delay cube.
        names = [
            "cfr_by_pair_rx_element" if cfr else "slow_time_by_pair_rx_element",
            "range_profile_by_pair",
        ]
        for name in names:
            np.testing.assert_array_equal(row[name], dense[index][name])
        if "result" in row["temporal"]:
            np.testing.assert_array_equal(
                row["range_doppler_power_by_pair"], dense[index]["range_doppler_power_by_pair"]
            )
            assert (
                row["temporal"]["acquisition"]["acquisition_config_id"]
                != dense[index]["temporal"]["acquisition"]["acquisition_config_id"]
            )


def test_gap_rejected_without_advancing_hop_or_noise_state():
    processor = _processor(4)
    pulse = PulsePaths([1 + 0j], [1.37 / 100e6])
    processor.process(None, 0, pulse, source_time_s=0.0)
    history = np.array(processor._fast_history_raw)
    noise = deepcopy(processor._noise_rng.bit_generator.state)
    with pytest.raises(ValueError, match="missing pulse"):
        processor.process(None, 2, pulse, source_time_s=2 / PRF)
    assert processor._source_sample_count == 1
    np.testing.assert_array_equal(processor._fast_history_raw, history)
    assert processor._noise_rng.bit_generator.state == noise


def test_full_8192_pulse_stream_publishes_exactly_125_native_windows():
    processor = _processor(64, samples=256)
    published = []
    for index in range(8192):
        result = processor.process(
            None,
            index,
            PulsePaths([np.exp(2j * np.pi * 375 * index / PRF)], [1.37 / 100e6]),
            source_time_s=index / PRF,
        )
        if result["rd_fresh"]:
            published.append(index)
            assert result["cfr_by_pair_rx_element"].shape[-2] == 256
    assert published == list(range(255, 8192, 64))
    assert len(published) == 125
    assert processor._source_sample_count == 8192
    np.testing.assert_array_equal(processor._source_time_history, np.arange(7936, 8192) / PRF)


def test_detection_and_tracker_update_only_at_published_cpi_centers(monkeypatch):
    import generator.core.sensing.strategies._shared_helpers as helpers

    processor = _processor(
        4,
        range_response=RangeResponseConfig(),
        coherent_clutter_mode="legacy",
        output_detections=True,
        track_confirmation_enabled=True,
        cfar_guard_cells=0,
        cfar_ref_cells=1,
        cfar_rd_guard_cells_doppler=0,
        cfar_rd_guard_cells_range=0,
        cfar_rd_ref_cells_doppler=1,
        cfar_rd_ref_cells_range=1,
    )
    range_calls, detector_times, tracker_times = [], [], []
    original_range = helpers.cfar_1d
    original_detector = processor._run_2d_cfar
    original_tracker = processor._tracker.filter_detections

    def range_detector(*args, **kwargs):
        range_calls.append(processor._source_sample_count)
        return original_range(*args, **kwargs)

    def detector(*args, **kwargs):
        detector_times.append(kwargs["update_time_s"])
        return original_detector(*args, **kwargs)

    def tracker(*args, **kwargs):
        tracker_times.append(kwargs["update_time_s"])
        return original_tracker(*args, **kwargs)

    monkeypatch.setattr(helpers, "cfar_1d", range_detector)
    monkeypatch.setattr(processor, "_run_2d_cfar", detector)
    monkeypatch.setattr(processor._tracker, "filter_detections", tracker)
    rows = _run(processor, 24)
    expected = list(range(7, 24, 4))
    assert range_calls == expected
    assert detector_times == tracker_times
    np.testing.assert_allclose(tracker_times, (np.array(expected) - 3.5) / PRF, rtol=0, atol=1e-15)
    assert [i for i, row in enumerate(rows) if "detections" in row] == expected


def test_native_hop_packed_roundtrip_keeps_complete_unprocessed_pulses(tmp_path):
    from dataclasses import replace

    from shared.frames.frame_set_writer import FrameSetWriter
    from shared.sensing.loader import load_frames_sensing
    from shared.sensing.postprocessor import PostProcessingConfig, SensingPostProcessor
    from tests.shared.test_sensing_postprocessor import _make_standard_frame

    rows = _run(_processor(4), 16)
    source = tmp_path / "native"
    with FrameSetWriter.create_new(source, compression="lzf") as writer:
        for index, sensing in enumerate(rows):
            writer.append(
                replace(
                    _make_standard_frame(frame_idx=index, sensing=sensing), timestamp_s=index / PRF
                )
            )
        writer.finalize()
    loaded = load_frames_sensing(source)
    assert len(loaded) == len(rows)
    for index, frame in enumerate(loaded):
        saved = frame["_typed_sensing"]
        assert saved["config_resolved"]["acquisition"]["cpi_hop_samples"] == 4
        assert saved["config_resolved"]["processing"]["result_stride_samples"] == 4
        np.testing.assert_array_equal(
            saved["cfr_by_pair_rx_element"], rows[index]["cfr_by_pair_rx_element"]
        )
        if index not in (7, 11, 15):
            assert "result" not in saved["temporal"]
            assert "display_binding" not in saved["temporal"]
            assert "truth" not in saved["temporal"]
            if index >= 7:
                assert saved["temporal"]["acquisition"]["completeness"] == "complete"
            assert not any(
                d["kind"] in {"range_doppler", "micro_doppler"} for d in saved["products"].values()
            )
            continue
        replay = SensingPostProcessor(
            PostProcessingConfig(skip_detection=True)
        ).process_from_frequency_observation(saved)
        np.testing.assert_allclose(
            abs(replay["rd_cube"]) ** 2,
            rows[index]["range_doppler_power_by_pair"][0],
            rtol=2e-13,
            atol=2e-13,
        )
        np.testing.assert_array_equal(
            replay["cfr_rebuilt"]["range_profile"], rows[index]["range_profile_by_pair"][0]
        )


@pytest.mark.parametrize(
    "samples,hop,count", [(8, 3, 25), (16, 3, 36), (256, 128, 512), (256, 256, 768)]
)
def test_saved_stft_columns_have_the_actual_native_publication_cadence(samples, hop, count):
    rows = _run(_processor(hop, samples=samples), count)
    times, widths = [], []
    display_hop = ((max(1, samples // 4) + hop - 1) // hop) * hop
    for row in rows:
        if "micro_doppler_by_pair" not in row:
            continue
        power = row["micro_doppler_by_pair"]
        assert power.shape == (1, 1, samples)
        descriptor = row["products"]["micro_doppler_by_pair"]
        processing = descriptor["processing"]
        assert processing["hop_samples"] == display_hop
        assert processing["recipe"]["hop_samples"] == display_hop
        assert processing["recipe"]["window_samples"] == samples
        times.extend(row["micro_doppler_time_s"])
        widths.extend([processing["hop_samples"] / PRF] * power.shape[1])
    ends = np.arange(samples - 1, count, display_hop)
    assert len(times) == len(ends)
    np.testing.assert_allclose(times, (ends - (samples - 1) / 2) / PRF, rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.diff(times), widths[:-1], rtol=0, atol=1e-15)


@pytest.mark.parametrize("display", [{"hop_samples": 1}, {"hop_samples": 5}, {"window_samples": 3}])
def test_explicit_display_cadence_must_fit_native_result_grid(display):
    with pytest.raises(ConfigurationError, match="recording.display.*cpi_hop_samples"):
        _processor(4, recording_display=display)


def test_explicit_display_hop_can_skip_native_results_without_fabricating_columns():
    rows = _run(_processor(2, recording_display={"hop_samples": 4}), 24)
    assert [i for i, row in enumerate(rows) if "result" in row["temporal"]] == list(range(7, 24, 2))
    assert [i for i, row in enumerate(rows) if "micro_doppler_by_pair" in row] == list(
        range(7, 24, 4)
    )
