from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from generator.core.configuration import SensingConfig
from generator.core.sensing.acquisition import (
    ANALYTIC_ORACLE_MODE,
    resolve_sensing_execution_values,
)
from generator.core.sensing.processor import create_sensing_processor
from shared.sensing.detection.angle import AnglePeakDetectionConfig
from shared.sensing.detection.cfar import cluster_detections_2d
from shared.sensing.detection.pipeline import (
    DetectionPipelineConfig,
    _compute_snr_map,
    apply_detection_range_gate,
    apply_ghost_filter,
    run_angle_enhanced_detection_pipeline,
    run_detection_pipeline,
    run_vote_fusion_pipeline,
)
from shared.sensing.detection.vote import cluster_detections_3d
from shared.sensing.postprocessor import PostProcessingConfig, SensingPostProcessor
from shared.sensing.signal_processing import C0, RangeDopplerResult


def test_dynamic_range_is_an_explicit_threshold_with_a_weak_target_tradeoff():
    power = np.full((32, 64), 1e-12)
    power[10, 12], power[16, 30], power[22, 50] = 1, 1e-4, 1e-7
    rd = RangeDopplerResult(power, np.arange(64.0), np.arange(32.0), None)
    config = DetectionPipelineConfig((1, 1), (3, 3), 3, min_snr_db=6)
    legacy = run_detection_pipeline(rd, config=config, wavelength_m=1)
    limited = run_detection_pipeline(
        rd,
        config=replace(config, cfar_max_dynamic_range_db=60),
        wavelength_m=1,
        return_stage_artifacts=True,
    )
    assert len(legacy.detections) == 3
    # A real echo 40 dB down survives; one 70 dB down does not. This is an
    # operating-point limit, not proof that every suppressed peak was false.
    assert {d["range_bin"] for d in limited.detections} == {12, 30}
    stages = limited.stages
    assert stages.cfar_threshold[22, 50] == pytest.approx(1e-6)
    np.testing.assert_array_equal(stages.cfar_input_power, power)
    np.testing.assert_array_equal(
        stages.cfar_mask, (power > stages.cfar_threshold) & stages.cfar_eligible_mask
    )
    for scale in (1e-10, 1e10):
        scaled = run_detection_pipeline(
            replace(rd, power_linear=power * scale),
            config=replace(config, cfar_max_dynamic_range_db=60),
            wavelength_m=1,
        )
        assert {d["range_bin"] for d in scaled.detections} == {12, 30}


@pytest.mark.parametrize("value", [True, -1, float("inf"), float("nan"), "60"])
def test_dynamic_range_configuration_rejects_malformed_values(value):
    with pytest.raises(ValueError):
        DetectionPipelineConfig((1, 1), (3, 3), 3, cfar_max_dynamic_range_db=value)
    with pytest.raises(ValueError):
        PostProcessingConfig(cfar_max_dynamic_range_db=value)


def _make_rd_result() -> RangeDopplerResult:
    fft_d = 16
    fft_r = 32
    prf_hz = 16.0
    # A declared positive background makes the CFAR noise denominator and SNR
    # scientifically defined; sparse zero-only training windows are tested
    # separately below.
    rd_map = np.full((fft_d, fft_r), 1e-3, dtype=np.float64)
    rd_map[10, 12] = 100.0
    rd_map[10, 13] = 30.0

    range_axis = np.arange(fft_r, dtype=np.float64)
    doppler_axis = np.fft.fftshift(np.fft.fftfreq(fft_d, d=1.0 / prf_hz))
    return RangeDopplerResult(
        power_linear=rd_map,
        range_axis_m=range_axis,
        doppler_axis_hz=doppler_axis,
        cube=None,
    )


@pytest.mark.parametrize(
    "estimation_api",
    [
        cluster_detections_2d,
        cluster_detections_3d,
        run_detection_pipeline,
        run_vote_fusion_pipeline,
        run_angle_enhanced_detection_pipeline,
    ],
)
def test_estimation_apis_cannot_accept_target_truth(estimation_api: object) -> None:
    """Target positions are not a valid input to any detection-stage API."""
    assert "target_positions" not in inspect.signature(estimation_api).parameters


def test_processor_and_postprocessor_detect_same_targets_from_same_rd_map() -> None:
    rd_result = _make_rd_result()
    cfg = SensingConfig(
        enabled=True,
        output_range_profile=False,
        output_range_doppler=False,
        output_detections=True,
        bandwidth=C0,
        prf_hz=16.0,
        fft_size_range=32,
        fft_size_doppler=16,
        range_mode="bistatic",
        persist_detection_stage_artifacts=True,
    )
    cfg.processing_mode = ANALYTIC_ORACLE_MODE
    resolved = resolve_sensing_execution_values(
        cfg,
        cir_time_steps=1,
        cir_sampling_frequency_hz=None,
        frame_dt_s=1.0 / cfg.prf_hz,
    )
    processor = create_sensing_processor(cfg, C0, resolved)

    sensing_data: dict[str, object] = {}
    processor._run_2d_cfar(rd_result, sensing_data, update_time_s=0.0)
    processor_dets = sensing_data.get("detections")
    assert isinstance(processor_dets, list)
    assert "range_doppler_cfar_mask" not in sensing_data
    assert sensing_data["range_doppler_cfar_mask_by_pair"].shape == (1, 16, 32)
    assert sensing_data["range_doppler_cfar_mask_by_pair"].dtype == np.bool_

    postprocessor = SensingPostProcessor(PostProcessingConfig())
    post_result = postprocessor.process_from_rd_map(
        rd_result.power_linear,
        {
            "config": {
                "fft_size_range": 32,
                "fft_size_doppler": 16,
                "sampling_frequency": C0,
                "prf_hz": 16.0,
                "range_mode": "bistatic",
                "carrier_freq_hz": C0,
            }
        },
    )
    post_dets = post_result["detections"]

    assert len(processor_dets) == len(post_dets) == 1
    assert processor_dets[0]["range_m"] == pytest.approx(post_dets[0]["range_m"])
    assert processor_dets[0]["velocity_m_s"] == pytest.approx(post_dets[0]["velocity_m_s"])
    assert processor_dets[0]["doppler_bin"] == post_dets[0]["doppler_bin"]
    assert processor_dets[0]["range_bin"] == post_dets[0]["range_bin"]


@pytest.mark.parametrize("tracker_type", ["m_of_n", "kalman", "jpdaf"])
def test_rd_only_postprocessing_tracks_without_fabricated_angles(tracker_type) -> None:
    rd = _make_rd_result()
    processor = SensingPostProcessor(
        PostProcessingConfig(tracker_type=tracker_type, tracker_m=1, tracker_n=1)
    )
    for timestamp in (0.0, 0.1):
        result = processor.process_from_rd_map(
            rd.power_linear,
            {
                "config": {
                    "fft_size_range": 32,
                    "fft_size_doppler": 16,
                    "sampling_frequency": C0,
                    "prf_hz": 16.0,
                    "range_mode": "bistatic",
                    "carrier_freq_hz": C0,
                }
            },
            update_time_s=timestamp,
        )
        assert len(result["detections"]) == 1
        detection = result["detections"][0]
        assert detection["track_id"] == 1
        assert "aoa_az_deg" not in detection
        assert "aoa_el_deg" not in detection


def test_detection_pipeline_stage_artifacts_include_cluster_labels_and_power_db() -> None:
    rd_result = _make_rd_result()
    result = run_detection_pipeline(
        rd_result,
        config=DetectionPipelineConfig(
            cfar_guard_cells=(1, 2),
            cfar_reference_cells=(4, 8),
            cfar_threshold_scale=3.0,
            min_snr_db=0.0,
        ),
        wavelength_m=1.0,
        return_stage_artifacts=True,
    )

    assert result.stages is not None
    assert result.stages.cfar_input_power.shape == rd_result.power_linear.shape
    assert result.stages.cfar_input_power_db.shape == rd_result.power_linear.shape
    assert result.stages.cfar_threshold.shape == rd_result.power_linear.shape
    assert result.stages.cfar_training_cell_count.shape == rd_result.power_linear.shape
    assert result.stages.cfar_eligible_mask.shape == rd_result.power_linear.shape
    assert result.stages.snr_valid_mask.shape == rd_result.power_linear.shape
    assert result.stages.cluster_labels.shape == rd_result.power_linear.shape
    assert np.max(result.stages.cluster_labels) >= 0
    assert np.isfinite(result.stages.cfar_input_power_db).all()
    assert result.detections
    assert result.metadata["cfar"]["threshold_semantics"] == ("raw_linear_power_multiplier")
    assert result.metadata["cfar"]["edge_policy"] == "full_window"
    assert result.metadata["snr"]["invalid_cell_policy"].startswith("reject")
    detection = result.detections[0]
    assert detection["path_closing_rate_m_s"] == pytest.approx(detection["doppler_hz"])
    assert detection["monostatic_equivalent_velocity_m_s"] == pytest.approx(
        detection["path_closing_rate_m_s"] / 2.0
    )
    assert detection["velocity_m_s"] == pytest.approx(
        detection["monostatic_equivalent_velocity_m_s"]
    )


def test_detection_pipeline_rejects_zero_noise_snr_instead_of_inventing_large_db() -> None:
    rd_result = _make_rd_result()
    rd_result.power_linear.fill(0.0)
    rd_result.power_linear[10, 12] = 100.0

    result = run_detection_pipeline(
        rd_result,
        config=DetectionPipelineConfig(
            cfar_guard_cells=(1, 1),
            cfar_reference_cells=(2, 2),
            cfar_threshold_scale=2.0,
            min_snr_db=-100.0,
        ),
        wavelength_m=1.0,
        return_stage_artifacts=True,
    )

    assert result.stages is not None
    assert result.stages.cfar_mask[10, 12]
    assert not result.stages.snr_valid_mask[10, 12]
    assert result.stages.snr_map[10, 12] == 0.0
    assert result.detections == []
    assert result.metadata["snr_valid_cfar_cells"] == 0


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"cfar_guard_cells": (1.0, 1)}, "cfar_guard_cells"),
        ({"cfar_reference_cells": (1, 0)}, "cfar_reference_cells"),
        ({"cfar_threshold_scale": np.nan}, "cfar_threshold_scale"),
        ({"cfar_method": "CA"}, "cfar_method"),
        ({"cfar_os_rank": 1.1}, "cfar_os_rank"),
        ({"cfar_edge_policy": "wrap"}, "cfar_edge_policy"),
        ({"nms_suppression_radius": (-1, 1)}, "nms_suppression_radius"),
        ({"cluster_range_gate_m": 0.0}, "cluster_range_gate_m"),
        ({"cluster_velocity_gate_m_s": np.inf}, "cluster_velocity_gate_m_s"),
        ({"min_snr_db": np.nan}, "min_snr_db"),
        ({"min_detection_range_m": -1.0}, "min_detection_range_m"),
        (
            {"min_detection_range_m": 2.0, "max_detection_range_m": 1.0},
            "must not exceed",
        ),
        ({"ghost_filter_velocity_gate_m_s": 0.0}, "ghost_filter_velocity_gate_m_s"),
        ({"ghost_filter_azimuth_gate_deg": 10.0}, "requires"),
    ],
)
def test_detection_pipeline_config_rejects_ambiguous_scientific_values(
    overrides: dict[str, object],
    match: str,
) -> None:
    values: dict[str, object] = {
        "cfar_guard_cells": (1, 1),
        "cfar_reference_cells": (2, 2),
        "cfar_threshold_scale": 2.0,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=match):
        DetectionPipelineConfig(**values)


def test_detection_pipeline_rejects_complex_power_instead_of_discarding_phase() -> None:
    rd_result = _make_rd_result()
    complex_power = rd_result.power_linear.astype(np.complex128)

    with pytest.raises(ValueError, match="real numeric linear-power"):
        run_detection_pipeline(
            rd_result,
            config=DetectionPipelineConfig(
                cfar_guard_cells=(1, 1),
                cfar_reference_cells=(2, 2),
                cfar_threshold_scale=2.0,
            ),
            wavelength_m=1.0,
            detection_power_map=complex_power,
        )


def test_detection_pipeline_validates_axes_before_an_empty_detection_return() -> None:
    rd_result = _make_rd_result()
    rd_result.power_linear.fill(0.0)
    rd_result.range_axis_m[4] = rd_result.range_axis_m[3]

    with pytest.raises(ValueError, match="strictly increasing"):
        run_detection_pipeline(
            rd_result,
            config=DetectionPipelineConfig(
                cfar_guard_cells=(1, 1),
                cfar_reference_cells=(2, 2),
                cfar_threshold_scale=2.0,
            ),
            wavelength_m=1.0,
        )


def test_snr_map_rejects_numeric_eligibility_coercion() -> None:
    with pytest.raises(ValueError, match="boolean"):
        _compute_snr_map(
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2), dtype=np.uint8),
        )


@pytest.mark.parametrize(
    ("function", "kwargs", "match"),
    [
        (
            apply_detection_range_gate,
            {"detections": [{}], "min_detection_range_m": 1.0},
            "missing range_m",
        ),
        (
            apply_detection_range_gate,
            {
                "detections": [{"range_m": np.nan}],
                "max_detection_range_m": 10.0,
            },
            "finite",
        ),
        (
            apply_ghost_filter,
            {"detections": [], "velocity_gate_m_s": -1.0},
            "positive",
        ),
        (
            apply_ghost_filter,
            {"detections": [{}], "velocity_gate_m_s": 1.0},
            "require range_m",
        ),
    ],
)
def test_detection_filters_fail_closed_on_missing_or_nonphysical_values(
    function: object,
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        function(**kwargs)


def test_ghost_filter_keeps_same_velocity_targets_when_azimuths_differ() -> None:
    detections = [
        {"range_m": 5.0, "velocity_m_s": 1.0, "aoa_az_deg": -25.0},
        {"range_m": 8.0, "velocity_m_s": 1.2, "aoa_az_deg": 22.0},
    ]

    filtered, metadata = apply_ghost_filter(
        detections,
        velocity_gate_m_s=3.0,
        azimuth_gate_deg=10.0,
    )

    assert len(filtered) == 2
    assert metadata["enabled"] is True
    assert metadata["azimuth_gate_deg"] == pytest.approx(10.0)


def test_ghost_filter_suppresses_far_detection_when_azimuths_match() -> None:
    detections = [
        {"range_m": 5.0, "velocity_m_s": 1.0, "aoa_az_deg": -5.0},
        {"range_m": 8.0, "velocity_m_s": 1.1, "aoa_az_deg": 1.0},
    ]

    filtered, _ = apply_ghost_filter(
        detections,
        velocity_gate_m_s=3.0,
        azimuth_gate_deg=10.0,
    )

    assert len(filtered) == 1
    assert filtered[0]["range_m"] == pytest.approx(5.0)


def test_angle_enhanced_pipeline_splits_one_rd_peak_into_two_azimuth_targets() -> None:
    rd_result = _make_rd_result()
    azimuth_cube = np.zeros((8, *rd_result.power_linear.shape), dtype=np.float64)
    # Add a low noise floor so the azimuth profile median is well below the
    # two peaks.  Without this, the profile has only 2 positive values and
    # the median equals ~97.5, causing the secondary peak to fail the
    # profile-SNR gate (secondary_peak_min_profile_snr_db default 6 dB).
    azimuth_cube[:, 10, 12] = 1.0
    azimuth_cube[1, 10, 12] = 100.0
    azimuth_cube[6, 10, 12] = 95.0
    azimuth_axis_deg = np.linspace(-50.0, 50.0, azimuth_cube.shape[0], dtype=np.float64)

    result = run_angle_enhanced_detection_pipeline(
        rd_result,
        azimuth_cube=azimuth_cube,
        azimuth_axis_deg=azimuth_axis_deg,
        config=DetectionPipelineConfig(
            cfar_guard_cells=(1, 1),
            cfar_reference_cells=(1, 1),
            cfar_threshold_scale=1.5,
            nms_suppression_radius=(1, 1),
            cluster_range_gate_m=1.0,
            cluster_velocity_gate_m_s=1.0,
            min_snr_db=0.0,
        ),
        angle_config=AnglePeakDetectionConfig(
            max_peaks_per_rd_cell=2,
            peak_relative_floor_db=10.0,
            peak_nms_radius_bins=1,
        ),
        wavelength_m=1.0,
        return_stage_artifacts=True,
    )

    assert result.metadata["detection_mode"] == "rd_angle_peaks_3d"
    assert len(result.detections) == 2
    azimuth_bins = sorted(det["azimuth_bin"] for det in result.detections)
    assert azimuth_bins == [1, 6]
    assert result.detections[0]["x_m"] == pytest.approx(
        result.detections[0]["range_m"] * np.cos(np.radians(result.detections[0]["aoa_az_deg"]))
    )
    assert result.detections[0]["path_closing_rate_m_s"] == pytest.approx(
        result.detections[0]["doppler_hz"]
    )
    assert result.detections[0]["velocity_m_s"] == pytest.approx(
        result.detections[0]["monostatic_equivalent_velocity_m_s"]
    )
    assert result.stages is not None
    assert result.stages.extra_artifacts is not None
    assert result.stages.extra_artifacts["angle_candidate_mask"].shape == azimuth_cube.shape


def test_angle_enhanced_pipeline_suppresses_weaker_nearby_sidelobe() -> None:
    rd_result = _make_rd_result()
    azimuth_cube = np.zeros((8, *rd_result.power_linear.shape), dtype=np.float64)
    azimuth_cube[2, 10, 12] = 100.0
    azimuth_cube[4, 10, 12] = 10.0
    azimuth_axis_deg = np.linspace(-50.0, 50.0, azimuth_cube.shape[0], dtype=np.float64)

    result = run_angle_enhanced_detection_pipeline(
        rd_result,
        azimuth_cube=azimuth_cube,
        azimuth_axis_deg=azimuth_axis_deg,
        config=DetectionPipelineConfig(
            cfar_guard_cells=(1, 1),
            cfar_reference_cells=(1, 1),
            cfar_threshold_scale=1.5,
            nms_suppression_radius=(1, 1),
            cluster_range_gate_m=1.0,
            cluster_velocity_gate_m_s=1.0,
            min_snr_db=0.0,
        ),
        angle_config=AnglePeakDetectionConfig(
            max_peaks_per_rd_cell=2,
            peak_relative_floor_db=20.0,
            peak_nms_radius_bins=0,
            sidelobe_suppression_enabled=True,
            sidelobe_azimuth_tolerance_bins=2,
            sidelobe_min_power_ratio_db=6.0,
        ),
        wavelength_m=1.0,
    )

    assert len(result.detections) == 1
    assert result.detections[0]["azimuth_bin"] == 2


def test_angle_enhanced_pipeline_requires_second_peak_to_be_close_in_strength() -> None:
    rd_result = _make_rd_result()
    azimuth_cube = np.zeros((8, *rd_result.power_linear.shape), dtype=np.float64)
    azimuth_cube[1, 10, 12] = 100.0
    azimuth_cube[6, 10, 12] = 20.0
    azimuth_axis_deg = np.linspace(-50.0, 50.0, azimuth_cube.shape[0], dtype=np.float64)

    result = run_angle_enhanced_detection_pipeline(
        rd_result,
        azimuth_cube=azimuth_cube,
        azimuth_axis_deg=azimuth_axis_deg,
        config=DetectionPipelineConfig(
            cfar_guard_cells=(1, 1),
            cfar_reference_cells=(1, 1),
            cfar_threshold_scale=1.5,
            nms_suppression_radius=(1, 1),
            cluster_range_gate_m=1.0,
            cluster_velocity_gate_m_s=1.0,
            min_snr_db=0.0,
        ),
        angle_config=AnglePeakDetectionConfig(
            max_peaks_per_rd_cell=2,
            peak_relative_floor_db=10.0,
            peak_nms_radius_bins=1,
            secondary_peak_max_gap_db=3.0,
        ),
        wavelength_m=1.0,
    )

    assert len(result.detections) == 1
    assert result.detections[0]["azimuth_bin"] == 1


def test_angle_enhanced_pipeline_can_ignore_edge_azimuth_bins() -> None:
    rd_result = _make_rd_result()
    azimuth_cube = np.zeros((8, *rd_result.power_linear.shape), dtype=np.float64)
    azimuth_cube[0, 10, 12] = 100.0
    azimuth_cube[3, 10, 12] = 95.0
    azimuth_axis_deg = np.linspace(-90.0, 90.0, azimuth_cube.shape[0], dtype=np.float64)

    result = run_angle_enhanced_detection_pipeline(
        rd_result,
        azimuth_cube=azimuth_cube,
        azimuth_axis_deg=azimuth_axis_deg,
        config=DetectionPipelineConfig(
            cfar_guard_cells=(1, 1),
            cfar_reference_cells=(1, 1),
            cfar_threshold_scale=1.5,
            nms_suppression_radius=(1, 1),
            cluster_range_gate_m=1.0,
            cluster_velocity_gate_m_s=1.0,
            min_snr_db=0.0,
        ),
        angle_config=AnglePeakDetectionConfig(
            max_peaks_per_rd_cell=2,
            peak_relative_floor_db=10.0,
            peak_nms_radius_bins=1,
            edge_exclusion_bins=1,
        ),
        wavelength_m=1.0,
    )

    assert len(result.detections) == 1
    assert result.detections[0]["azimuth_bin"] == 3
