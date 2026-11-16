"""Retention changes disk products without changing generated measurements."""

import numpy as np
import pytest

from generator.core.configuration import SensingConfig
from generator.core.exceptions import ConfigurationError
from shared.extensions.sensing import validate_sensing_contract
from shared.scenarios.sensing_groups import normalize_sensing_groups
from shared.sensing.cfr_replay import rebuild_frequency_observation
from shared.sensing.products import validate_product_descriptors
from tests.generator.unit.test_sensing_cfr_lifecycle import native_cfr


@pytest.mark.parametrize("profile", ["results", "maps", "reprocessing", "full"])
def test_profiles_keep_measurements_and_native_contract(profile):
    args = dict(
        elements=4,
        rows=2,
        detection=True,
        direction=(0.25, 0.2),
        array_mode="beam_scan",
        clutter="none",
    )
    reference = native_cfr(**args)
    value = native_cfr(**args, recording=profile)
    assert value["detections"] == reference["detections"]
    assert not validate_product_descriptors(value, require_described_data=True)
    assert not validate_sensing_contract({"sensing": value}, require_schema_version=True)
    canonical = [
        key for key in ("cfr_by_pair_rx_element", "slow_time_by_pair_rx_element") if key in value
    ]
    assert canonical == (["cfr_by_pair_rx_element"] if profile in {"reprocessing", "full"} else [])
    assert "cir_by_pair_rx_element" not in value
    assert value["recording"]["profile"] == profile
    if profile != "results":
        np.testing.assert_array_equal(
            value["range_doppler_power_by_pair"], abs(reference["range_doppler_by_pair"]) ** 2
        )
        np.testing.assert_array_equal(
            value["delay_power_by_pair"],
            abs(reference["slow_time_by_pair_rx_element"].sum(axis=1) / 2) ** 2,
        )
        assert "micro_doppler_by_pair" in value
        assert "range_doppler_raw_power_by_pair" not in value


@pytest.mark.parametrize("profile", ["results", "maps", "reprocessing", "full"])
@pytest.mark.parametrize("model", ["point_bin_floor", "cfr_ifft"])
def test_array_rd_mode_preserves_its_computed_angular_display(profile, model):
    args = dict(elements=4, rows=2, array_mode="rd", detection=True, model=model)
    reference = native_cfr(**args)
    value = native_cfr(**args, recording=profile)
    np.testing.assert_equal(value["detections"], reference["detections"])
    key = "range_doppler_angle_by_pair"
    assert (key in value) == (profile != "results")
    if profile != "results":
        np.testing.assert_array_equal(value[key], reference[key])
        assert value["products"][key]["kind"] == "range_doppler_angle"
        assert value["products"][key]["representation"] == "power_linear"
    assert "range_doppler_azimuth_cube" not in value
    assert not validate_sensing_contract({"sensing": value}, require_schema_version=True)


def test_results_without_diagnostics_has_no_numeric_product():
    value = native_cfr(recording="results")
    assert not value["products"]
    assert not any(isinstance(item, np.ndarray) for item in value.values())


@pytest.mark.parametrize("profile", ["results", "maps", "reprocessing", "full"])
@pytest.mark.parametrize("tracker", ["cartesian", "cartesian3d"])
def test_disabled_cartesian_tracking_preserves_unlocalized_measurements(profile, tracker):
    reference = native_cfr(detection=True, tracker=tracker, tracking=False)
    value = native_cfr(detection=True, tracker=tracker, tracking=False, recording=profile)
    np.testing.assert_equal(value["detections"], reference["detections"])
    assert value["detections_before_tracking"]
    assert (
        value["detections_before_tracking_metadata"]["coordinate_frame"] == "receiver_measurement"
    )
    for report in value["detections_before_tracking"]:
        assert "range_m" in report and "velocity_m_s" in report
        assert not {"x_m", "y_m", "z_m", "aoa_az_deg", "aoa_el_deg"} & report.keys()
    assert not validate_sensing_contract({"sensing": value}, require_schema_version=True)


def test_canonical_frequency_reconstructs_exact_delay_samples():
    args = dict(elements=4, rows=2, clutter="none", noise=0.01)
    complete = native_cfr(**args)
    retained = native_cfr(**args, recording="reprocessing")
    rebuilt = rebuild_frequency_observation(retained, mti_enabled=False)
    np.testing.assert_array_equal(rebuilt["delay"], complete["slow_time_by_pair_rx_element"][0])


def test_point_bin_retains_only_actual_delay_observation():
    value = native_cfr(recording="reprocessing", model="point_bin_floor")
    assert "slow_time_by_pair_rx_element" in value
    assert "cfr_by_pair_rx_element" not in value
    assert "cir_by_pair_rx_element" not in value


def test_grouped_profiles_reject_ambiguous_legacy_policy():
    with pytest.raises(ValueError, match="cannot be combined"):
        normalize_sensing_groups(
            {"recording": {"profile": "maps", "complex_observations": "always"}}
        )
    assert (
        normalize_sensing_groups({"recording": {"profile": "maps"}})["recording_profile"] == "maps"
    )
    with pytest.raises(ConfigurationError, match="recording.profile"):
        SensingConfig(recording_profile="typo")


def test_full_enables_diagnostic_recording():
    config = SensingConfig(recording_profile="full")
    assert config.persist_detection_stage_artifacts and config.persist_tracking_diagnostics


def test_display_hop_is_validated_against_effective_acquisition_window():
    with pytest.raises(ConfigurationError, match="Hop"):
        SensingConfig(
            recording_profile="maps", fft_size_doppler=16, recording_display={"hop_samples": 32}
        )


def test_results_beam_diagnostics_retain_their_source_even_with_legacy_rd_off():
    value = native_cfr(
        recording="results",
        output_rd=False,
        elements=4,
        rows=2,
        array_mode="beam_scan",
        detection=True,
    )
    assert "range_doppler_angle_by_pair" in value
    assert not validate_product_descriptors(value, require_described_data=True)


def test_streamed_display_matches_batch_recipe_and_emits_no_duplicate_columns():
    from shared.sensing.motion_analysis import MotionRecipe, compute_motion_spectra
    from shared.sensing.recording import add_display_products

    cfg = SensingConfig(recording_profile="maps", fft_size_doppler=8, prf_hz=1000)
    times = np.arange(20) / 1000
    ranges = np.arange(4, dtype=float)
    samples = np.exp(2j * np.pi * 150 * times[:, None]) * np.arange(1, 5)[None]
    last = None
    columns, centers = [], []
    for stop in range(1, 21):
        block = {}
        start = max(0, stop - 8)
        last = add_display_products(
            block,
            cfg,
            samples=samples[start:stop],
            times_s=times[start:stop],
            range_axis_m=ranges,
            sequential=True,
            last_spectrum_time_s=last,
        )
        assert block["delay_power_by_pair"].shape == (1, 1, 4)
        if "micro_doppler_by_pair" in block:
            columns.append(block["micro_doppler_by_pair"][0])
            centers.extend(block["micro_doppler_time_s"])
    batch = compute_motion_spectra(
        samples, times, ranges, MotionRecipe(range_max_m=3, window_samples=8, hop_samples=2)
    )
    # Local floating timestamp differences alter dt at machine precision.
    np.testing.assert_allclose(
        np.concatenate(columns), batch["removed_psd"].T, rtol=2e-14, atol=1e-25
    )
    np.testing.assert_array_equal(centers, batch["time_s"])


def test_display_gaps_are_explicit_and_cpi_calls_do_not_join():
    from shared.sensing.recording import add_display_products

    cfg = SensingConfig(recording_profile="maps", fft_size_doppler=8, prf_hz=1000)
    samples = np.ones((8, 4), dtype=complex)
    block = {}
    last = add_display_products(
        block,
        cfg,
        samples=samples,
        times_s=np.array([0, 1, 2, 3, 6, 7, 8, 9]) / 1000,
        range_axis_m=np.arange(4),
        sequential=True,
    )
    assert last is None and "micro_doppler_by_pair" not in block
    assert block["display_status"]["micro_doppler"] == "acquisition gap"
    for offset in (0, 0.2):
        block = {}
        add_display_products(
            block,
            cfg,
            samples=samples,
            times_s=offset + np.arange(8) / 1000,
            range_axis_m=np.arange(4),
        )
        assert block["micro_doppler_by_pair"].shape == (1, 1, 8)


def test_saved_stft_grid_is_stable_across_physical_cpi_epochs():
    from shared.sensing.recording import add_display_products

    cfg = SensingConfig(recording_profile="maps", fft_size_doppler=128, prf_hz=2000)
    expected = np.fft.fftshift(np.fft.fftfreq(128, d=1 / 2000))
    for index in range(40):
        block = {}
        add_display_products(
            block,
            cfg,
            samples=np.ones((128, 4), complex),
            times_s=index * 0.064 + np.arange(128) / 2000,
            range_axis_m=np.arange(4),
        )
        np.testing.assert_array_equal(block["micro_doppler_frequency_hz"], expected)


@pytest.mark.parametrize("profile", ["results", "maps", "reprocessing", "full"])
def test_profiles_roundtrip_through_existing_native_hdf5(tmp_path, profile):
    from shared.sensing.loader import load_frames_sensing
    from tests.shared.test_sensing_postprocessor import (
        _make_standard_frame,
        _write_packed_frame_set,
    )

    value = native_cfr(recording=profile, detection=True)
    root = tmp_path / profile
    _write_packed_frame_set(root, _make_standard_frame(frame_idx=0, sensing=value))
    loaded = load_frames_sensing(root)[0]["_typed_sensing"]
    assert loaded["recording"] == value["recording"]
    np.testing.assert_equal(loaded["detections"], value["detections"])
    assert set(loaded["products"]) == set(value["products"])
    for key in value["products"]:
        np.testing.assert_array_equal(loaded[key], value[key])


@pytest.mark.parametrize("profile", ["results", "maps", "reprocessing", "full"])
@pytest.mark.parametrize("model", ["point_bin_floor", "cfr_ifft"])
def test_sequential_profiles_keep_window_results_and_only_one_input(profile, model):
    from shared.sensing.range_response import RangeResponseConfig
    from tests.generator.unit.test_sensing_processor import _create_processor, _FakePaths

    def run(recording):
        cfg = SensingConfig(
            enabled=True,
            recording_profile=recording,
            processing_mode="sequential_stft",
            output_detections=False,
            output_range_profile=True,
            fft_size_range=32,
            fft_size_doppler=4,
            window_range="boxcar",
            raw_payload_policy="always",
            range_response=(
                RangeResponseConfig(model, 32) if model == "cfr_ifft" else RangeResponseConfig()
            ),
        )
        processor = _create_processor(cfg, carrier_frequency_hz=28e9)
        paths = _FakePaths([complex(np.exp(0.4j * i)) for i in range(12)], delay_s=20e-9)
        return [
            processor.process(None, frame_idx=i, paths=paths, source_time_s=i / cfg.prf_hz)
            for i in range(12)
        ]

    reference, retained = run(None), run(profile)
    canonical = "cfr_by_pair_rx_element" if model == "cfr_ifft" else "slow_time_by_pair_rx_element"
    for source, result in zip(reference[3:], retained[3:]):
        if profile != "results":
            np.testing.assert_array_equal(
                result["range_doppler_power_by_pair"], abs(source["range_doppler_by_pair"]) ** 2
            )
            assert result["delay_power_by_pair"].shape == (1, 1, 32)
            assert "micro_doppler_by_pair" in result
        if profile in {"reprocessing", "full"}:
            np.testing.assert_array_equal(result[canonical], source[canonical])
        else:
            assert canonical not in result


def test_delay_observation_preserves_actual_post_projection_noise():
    from shared.sensing.recording import canonical_observation_key
    from tests.generator.unit.test_sensing_processor import _create_processor, _FakePaths

    outputs = []
    for recording in (None, "reprocessing"):
        cfg = SensingConfig(
            enabled=True,
            recording_profile=recording,
            processing_mode="sequential_stft",
            output_detections=False,
            fft_size_range=16,
            fft_size_doppler=4,
            noise_enabled=True,
            noise_snr_db=10,
            rng_seed=42,
            raw_payload_policy="always",
        )
        assert canonical_observation_key(cfg) == "slow_time_by_pair_rx_element"
        processor = _create_processor(cfg, carrier_frequency_hz=28e9)
        paths = _FakePaths([1 + 1j] * 4, delay_s=20e-9)
        outputs.append(
            [
                processor.process(None, frame_idx=i, paths=paths, source_time_s=i / cfg.prf_hz)
                for i in range(4)
            ][-1]
        )
    np.testing.assert_array_equal(
        outputs[0]["slow_time_by_pair_rx_element"], outputs[1]["slow_time_by_pair_rx_element"]
    )
