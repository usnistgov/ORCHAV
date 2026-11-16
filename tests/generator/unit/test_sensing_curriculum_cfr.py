"""The authored sensing experiments keep a finite-bandwidth signal contract."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from generator import build_simulation_config
from generator.core.configuration.loader import resolve_sensing_document
from generator.core.sensing.acquisition import resolve_sensing_execution_values
from generator.core.sensing.processor import create_sensing_processor
from shared.scenarios import load_scenario_configuration
from shared.sensing.range_response import C0, RangeResponseConfig

ROOT = Path(__file__).resolve().parents[3]
CORE = (
    "cube_radar/stationary",
    "cube_radar/moving",
    "cube_radar/moving_tracking",
    "cube_radar/azimuth",
    "cube_radar/moving_tracking_3d",
    "cube_radar/two_moving",
    "cube_radar/oscillating_coherent",
    "cube_radar/oscillating_sequential",
    "foundations/static_and_moving_cubes",
)


@pytest.mark.parametrize("name", CORE)
def test_core_authored_and_runtime_range_models(name):
    directory = ROOT / "scenarios/sensing" / name
    authored = resolve_sensing_document(yaml.safe_load((directory / "scenario.yaml").read_text()))[
        "sensing"
    ]
    config = build_simulation_config(load_scenario_configuration(directory)).sensing
    assert authored["range_response"]["model"] == config.range_response.model == "cfr_ifft"
    assert authored["range_response"]["num_frequency_bins"] == 128
    assert authored["range_response"]["frequency_window"] == "hann"
    assert config.range_response.frequency_window == config.window_doppler == "hann"
    assert config.window_range == "boxcar"
    reprocessing = {
        "cube_radar/moving",
        "cube_radar/oscillating_coherent",
        "cube_radar/oscillating_sequential",
        "foundations/static_and_moving_cubes",
    }
    assert config.recording_profile == ("reprocessing" if name in reprocessing else "maps")
    assert "raw_payload_policy" not in authored
    assert config.bandwidth == 500e6
    period = C0 * config.range_response.num_frequency_bins / (2 * config.bandwidth)
    assert period == pytest.approx(38.373434624)
    assert config.max_detection_range_m is None or config.max_detection_range_m < period
    # This task changes authored choices, not the backward-compatible default.
    assert RangeResponseConfig.parse(None).model == "point_bin_floor"


@pytest.mark.parametrize(
    "name,samples,windows,hop",
    [
        ("stationary", 64, 1, 64),
        ("moving", 1024, 16, 64),
        ("two_moving", 1024, 16, 64),
        ("oscillating_coherent", 2048, 32, 64),
        ("oscillating_sequential", 2048, 1985, 1),
    ],
)
def test_cube_radar_acquisition_lengths_and_analysis_cadence(name, samples, windows, hop):
    directory = ROOT / "scenarios/sensing/cube_radar" / name
    scenario = load_scenario_configuration(directory)
    sim = build_simulation_config(scenario)
    assert scenario.timeline.steps == samples
    assert scenario.timeline.duration_s == pytest.approx((samples - 1) / 1000)
    assert sim.sensing.fft_size_range == sim.sensing.range_response.num_frequency_bins == 128
    assert sim.sensing.fft_size_doppler == 64
    assert sim.sensing.prf_hz == 1000
    assert sim.sensing.coherent_clutter_mode == "none"
    assert not sim.sensing.clutter_removal_enabled
    assert not sim.sensing.noise_enabled
    assert sim.sensing.range_response.frequency_noise.variance == 0
    execution = resolve_sensing_execution_values(
        sim.sensing,
        cir_time_steps=sim.cir_time_steps,
        cir_sampling_frequency_hz=sim.cir_sampling_frequency_hz,
        frame_dt_s=scenario.timeline.duration_s / (samples - 1),
    )
    assert execution.acquisition.cpi_length_samples == 64
    assert execution.acquisition.cpi_hop_samples == hop
    assert 1 + (samples - 64) // hop == windows


def test_oscillating_cube_comparison_keeps_scene_and_measurement_model_fixed():
    folder = ROOT / "scenarios/sensing/cube_radar"
    coherent, sequential = [
        yaml.safe_load((folder / name / "scenario.yaml").read_text())
        for name in ("oscillating_coherent", "oscillating_sequential")
    ]
    for key in ("scene", "actors", "timeline"):
        assert coherent[key] == sequential[key]
    for key in ("carrier_frequency_hz", "quality"):
        assert coherent["raytracing"][key] == sequential["raytracing"][key]
    sensing = [resolve_sensing_document(data)["sensing"] for data in (coherent, sequential)]
    for key in (
        "bandwidth",
        "prf_hz",
        "fft_size_range",
        "fft_size_doppler",
        "window_range",
        "window_doppler",
        "range_response",
        "coherent_clutter_mode",
    ):
        assert sensing[0][key] == sensing[1][key]


@pytest.mark.parametrize(
    "name,rd_override",
    [
        ("moving_tracking", False),
        ("azimuth", False),
        ("moving_tracking_3d", False),
        ("moving_tracking_3d", True),
    ],
)
def test_authored_detectors_on_known_complex_paths(name, rd_override):
    """No truth enters processing; assertions use independently prescribed paths."""
    directory = ROOT / "scenarios/sensing/cube_radar" / name
    sim = build_simulation_config(load_scenario_configuration(directory))
    config = sim.sensing
    # Inspect element phase for this numerical check; the authored maps profile
    # intentionally retains only the resulting power and detections.
    config.recording_profile = "reprocessing"
    if rd_override:
        config.array_detection_mode = "rd"
        config.beam_angle_oversampling = 1
    rows = sim.rx_antenna.num_rows if sim.rx_antenna else 1
    cols = sim.rx_antenna.num_cols if sim.rx_antenna else 1
    elements = rows * cols
    execution = resolve_sensing_execution_values(
        config,
        cir_time_steps=64,
        cir_sampling_frequency_hz=1000,
        frame_dt_s=0.001,
        rx_elements=elements,
    )
    processor = create_sensing_processor(config, 28e9, execution)
    uy, uz = 0.20, 0.15 if rows > 1 else 0.0
    steering = np.array(
        [np.exp(1j * np.pi * (col * uy - row * uz)) for col in range(cols) for row in range(rows)]
    )
    velocity = 1.6
    doppler = 2 * velocity * 28e9 / C0
    detections = []
    for index in range(6 if elements == 1 else 3):
        time = index * 0.064
        distance = 5.2 - velocity * time
        gains = (
            steering[:, None, None]
            * np.exp(2j * np.pi * doppler * (time + np.arange(64) / 1000))[None, None, :]
        )
        result = processor.process(
            SimpleNamespace(simulation_config=sim),
            index * 64,
            None,
            cir_stack=(gains[None, :, None, None], np.array([[[2 * distance / C0]]])),
            source_time_s=time,
        )
        # Detector measurements exist from the first CPI; tracking emits only
        # confirmed reports after the configured two-hit confirmation.
        measurements = result["detections_before_tracking"]
        assert len(measurements) == 1
        detection = measurements[0]
        if config.track_confirmation_enabled:
            assert len(result["detections"]) == (0 if index == 0 else 1)
        detections.extend(result["detections"])
        assert (
            abs(detection["range_m"] - distance)
            <= C0 * 128 / (2 * config.bandwidth * config.fft_size_range) / 2
        )
        assert abs(detection["velocity_m_s"] - velocity) <= C0 / (2 * 28e9) * 1000 / 64 / 2
        observation = result["cfr_by_pair_rx_element"][0]
        assert observation.shape == (elements, 64, 128)
        np.testing.assert_allclose(
            observation[:, 0, 0] / observation[0, 0, 0], steering, rtol=2e-13, atol=2e-13
        )
        if config.array_detection_mode == "beam_scan":
            expected_az = np.rad2deg(np.arctan2(uy, np.sqrt(1 - uy**2 - uz**2)))
            expected_el = np.rad2deg(np.arcsin(uz))
            assert abs(detection["aoa_az_deg"] - expected_az) < 2.0
            assert abs(detection["aoa_el_deg"] - expected_el) < 2.0
    if elements == 1:
        assert {d["track_id"] for d in detections} == {1}
        assert [d["track_confirmed"] for d in detections] == [True] * 5


@pytest.mark.parametrize(
    "name",
    [
        name
        for name in (
            "tracker_lab",
            "detection_stages",
            "detection_threshold_sweep",
            "two_target_resolution",
            "multipath_clutter_box",
        )
        if (ROOT / "scenarios/sensing/detection_tracking" / name / "scenario.yaml").is_file()
    ],
)
def test_historical_point_bin_controls_are_explicit(name):
    config = yaml.safe_load(
        (ROOT / "scenarios/sensing/detection_tracking" / name / "scenario.yaml").read_text()
    )
    assert (
        resolve_sensing_document(config)["sensing"]["range_response"]["model"] == "point_bin_floor"
    )
