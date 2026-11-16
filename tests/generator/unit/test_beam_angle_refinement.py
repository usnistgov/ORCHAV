"""Native beam refinement preserves detection support and precedes tracking."""

from functools import partial

import numpy as np
import pytest
from pydantic import ValidationError

from generator.core.configuration import SensingConfig
from generator.core.exceptions import ConfigurationError
from generator.core.sensing.acquisition import resolve_sensing_execution_values
from generator.core.sensing.processor import create_sensing_processor
from shared.scenarios.model import SensingModel
from shared.sensing.arrays import planar_spatial_fft
from shared.sensing.detection.beam_scan import refine_array_directions
from shared.sensing.signal_processing import RangeDopplerResult


@pytest.mark.parametrize("factor", [0, 17, 2.5, True])
def test_invalid_refinement_is_rejected_in_yaml_and_runtime(factor):
    with pytest.raises(ValidationError):
        SensingModel(array_detection_mode="beam_scan", beam_angle_oversampling=factor)
    with pytest.raises(ConfigurationError, match="beam_angle_oversampling"):
        SensingConfig(array_detection_mode="beam_scan", beam_angle_oversampling=factor)


def test_refinement_requires_beam_scan():
    with pytest.raises(ConfigurationError, match="requires array_detection_mode"):
        SensingConfig(beam_angle_oversampling=8)


def test_native_refinement_preserves_cfar_and_passes_refined_positions_to_tracker(monkeypatch):
    rng = np.random.default_rng(42)
    elements = 0.01 * (rng.normal(size=(32, 32, 48)) + 1j * rng.normal(size=(32, 32, 48)))
    elements[:, 20, 23] += np.exp(2j * np.pi * np.arange(32) * 0.5 * 0.34)
    spatial = planar_spatial_fft(
        elements,
        num_rows=1,
        num_cols=32,
        horizontal_spacing_wavelengths=0.5,
        vertical_spacing_wavelengths=0.5,
    )
    rd = RangeDopplerResult(
        abs(elements.sum(axis=0)) ** 2, np.arange(48.0), np.arange(32.0) - 16, None
    )
    outputs = []
    for factor in (1, 8):
        cfg = SensingConfig(
            enabled=True,
            range_mode="monostatic",
            processing_mode="coherent_cpi",
            array_detection_mode="beam_scan",
            beam_angle_oversampling=factor,
            fft_size_doppler=32,
            fft_size_range=48,
            cfar_rd_guard_cells_doppler=1,
            cfar_rd_guard_cells_range=1,
            cfar_rd_ref_cells_doppler=3,
            cfar_rd_ref_cells_range=3,
            cfar_min_snr_db=30,
            track_confirmation_enabled=False,
            persist_detection_stage_artifacts=True,
        )
        execution = resolve_sensing_execution_values(
            cfg,
            cir_time_steps=32,
            cir_sampling_frequency_hz=cfg.prf_hz,
            frame_dt_s=1 / cfg.prf_hz,
            rx_elements=32,
        )
        processor = create_sensing_processor(cfg, 3.5e9, execution)
        seen = []

        def tracker(detections, *, update_time_s):
            seen.extend(detections)
            return detections, {}

        monkeypatch.setattr(processor, "_apply_track_confirmation", tracker)
        payload = {}
        processor._run_2d_cfar(
            rd,
            payload,
            update_time_s=0.0,
            angular_power=abs(spatial.spectrum) ** 2,
            azimuth_direction_cosine=spatial.azimuth_direction_cosine,
            elevation_direction_cosine=spatial.elevation_direction_cosine,
            beam_angle_refiner=(
                partial(
                    refine_array_directions,
                    element_rd=elements,
                    num_rows=1,
                    num_cols=32,
                    horizontal_spacing_wavelengths=0.5,
                    vertical_spacing_wavelengths=0.5,
                    oversampling=factor,
                )
                if factor > 1
                else None
            ),
        )
        assert len(seen) == 1
        assert seen == payload["detections"]
        outputs.append(payload)
    coarse, refined = (p["detections"][0] for p in outputs)
    for key in (
        "range_m",
        "velocity_m_s",
        "power_linear",
        "beam_index",
        "range_bin",
        "doppler_bin",
    ):
        assert coarse[key] == refined[key]
    assert "coarse_aoa_az_deg" not in coarse
    assert refined["coarse_aoa_az_deg"] == coarse["aoa_az_deg"]
    true_angle = np.degrees(np.arcsin(0.34))
    assert abs(refined["aoa_az_deg"] - true_angle) < abs(coarse["aoa_az_deg"] - true_angle)
    np.testing.assert_array_equal(
        outputs[0]["range_doppler_cfar_mask_by_pair"], outputs[1]["range_doppler_cfar_mask_by_pair"]
    )
    assert outputs[1]["detection_filter_metadata"]["angle_estimation"]["oversampling"] == 8
