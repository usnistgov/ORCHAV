"""Regression checks for detection before fixed broadside combining."""

from dataclasses import replace

import numpy as np
import pytest

from shared.sensing.detection.beam_scan import run_beam_scan_detection_pipeline, visible_array_beams
from shared.sensing.detection.pipeline import DetectionPipelineConfig, run_detection_pipeline
from shared.sensing.signal_processing import RangeDopplerResult


def _inputs():
    beams = np.ones((3, 32, 48))
    # Nearby RD returns are separated in angle. A single-map NMS would
    # suppress one, even though the beams resolve both without target truth.
    beams[0, 15, 23] = 1000.0
    beams[2, 16, 23] = 500.0
    rd = RangeDopplerResult(
        power_linear=beams[1],
        range_axis_m=np.arange(48.0),
        doppler_axis_hz=np.arange(32.0) - 16,
        cube=None,
    )
    cfg = DetectionPipelineConfig(
        cfar_guard_cells=(1, 1),
        cfar_reference_cells=(3, 3),
        cfar_threshold_scale=3.0,
        min_snr_db=6.0,
        nms_suppression_radius=(2, 4),
        cluster_range_gate_m=1.0,
        cluster_velocity_gate_m_s=0.1,
    )
    return beams, rd, cfg


def test_off_broadside_targets_survive_beamwise_detection_and_tracker_runs_once():
    beams, rd, cfg = _inputs()
    updates = []

    def tracker(detections):
        updates.append(detections)
        return detections, {"calls": len(updates)}

    baseline = run_detection_pipeline(rd, config=cfg, wavelength_m=1.0)
    assert baseline.detections == []
    pooled = run_detection_pipeline(
        replace(rd, power_linear=beams.max(axis=0)), config=cfg, wavelength_m=1.0
    )
    assert len(pooled.detections) == 1
    result = run_beam_scan_detection_pipeline(
        rd,
        beam_power=beams,
        beam_azimuth_deg=np.array([-30.0, 0.0, 30.0]),
        beam_elevation_deg=np.zeros(3),
        config=cfg,
        wavelength_m=1.0,
        post_detection_filter=tracker,
        return_stage_artifacts=True,
    )
    assert len(updates) == 1
    assert len(result.detections) == 2
    assert sorted(d["aoa_az_deg"] for d in result.detections) == [-30.0, 30.0]
    assert all("x_m" in d and "y_m" in d for d in result.detections)
    # Projected noise is estimated within the selected beam, rather than on
    # a max-pooled map. Both bright cells have the unit training background.
    assert result.stages.cfar_noise_floor[15, 23] == pytest.approx(1.0)
    assert result.stages.cfar_noise_floor[16, 23] == pytest.approx(1.0)
    assert result.stages.nms_mask[15, 23] and result.stages.nms_mask[16, 23]


def test_bistatic_range_is_not_projected_as_a_monostatic_radius():
    beams, rd, cfg = _inputs()
    with pytest.raises(ValueError, match="requires monostatic"):
        run_beam_scan_detection_pipeline(
            rd,
            beam_power=beams,
            beam_azimuth_deg=np.array([-30.0, 0.0, 30.0]),
            beam_elevation_deg=np.zeros(3),
            config=cfg,
            wavelength_m=1.0,
            range_mode="bistatic",
        )


def test_dynamic_range_reference_is_separate_for_each_beam():
    beams, rd, cfg = _inputs()
    # The second beam can carry a much weaker target than the first. A global
    # strongest-beam floor would incorrectly remove its legitimate return.
    beams[:] = 1e-14
    beams[0, 12, 16] = 1.0
    beams[2, 21, 31] = 1e-8
    result = run_beam_scan_detection_pipeline(
        rd,
        beam_power=beams,
        beam_azimuth_deg=np.array([-30.0, 0.0, 30.0]),
        beam_elevation_deg=np.zeros(3),
        config=replace(cfg, cfar_max_dynamic_range_db=60),
        wavelength_m=1,
        return_stage_artifacts=True,
    )
    assert len(result.detections) == 2
    assert result.stages.cfar_threshold[12, 16] == pytest.approx(1e-6)
    assert result.stages.cfar_threshold[21, 31] < 1e-8


def test_pre_clutter_reference_rejects_tiny_residual_and_validates_beam_binding():
    beams, rd, cfg = _inputs()
    beams *= 1e-20
    az, el = np.array([-30.0, 0.0, 30.0]), np.zeros(3)
    reference = {
        "basis": "pre_clutter_spatial_fft_peak",
        "azimuth_deg": az.tolist(),
        "elevation_deg": el.tolist(),
        "power_linear": [1.0, 1.0, 1.0],
    }
    args = dict(
        beam_power=beams,
        beam_azimuth_deg=az,
        beam_elevation_deg=el,
        config=replace(cfg, cfar_max_dynamic_range_db=60),
        wavelength_m=1,
    )
    assert run_beam_scan_detection_pipeline(rd, **args).detections
    result = run_beam_scan_detection_pipeline(
        rd, **args, reference=reference, return_stage_artifacts=True
    )
    assert result.detections == []
    assert np.min(result.stages.cfar_threshold) >= 1e-6
    reference["azimuth_deg"].reverse()
    with pytest.raises(ValueError, match="exact visible beam coordinates"):
        run_beam_scan_detection_pipeline(rd, **args, reference=reference)


def test_same_rd_cell_has_one_candidate_and_empty_cpi_still_advances_tracker():
    beams, rd, cfg = _inputs()
    beams[:] = 1
    beams[:, 15, 23] = [1000.0, 500.0, 800.0]
    updates = []

    def tracker(detections):
        updates.append(len(detections))
        return detections, {}

    for beam_power in (beams, np.ones_like(beams)):
        result = run_beam_scan_detection_pipeline(
            rd,
            beam_power=beam_power,
            beam_azimuth_deg=np.array([-30.0, 0.0, 30.0]),
            beam_elevation_deg=np.zeros(3),
            config=cfg,
            wavelength_m=1.0,
            post_detection_filter=tracker,
        )
        assert len(result.detections) == (1 if len(updates) == 1 else 0)
    assert updates == [1, 0]


def test_upa_directions_use_joint_cosines_and_preserve_descending_elevation():
    cube = np.arange(6.0).reshape(2, 3, 1, 1)
    power, az, el = visible_array_beams(cube, np.array([0.5, 0.0]), np.array([-1.0, 0.0, 0.5]))
    assert power[:, 0, 0].tolist() == [1.0, 2.0, 4.0, 5.0]
    assert el.tolist() == pytest.approx([30.0, 30.0, 0.0, 0.0])
    assert az[1] == pytest.approx(np.degrees(np.arctan2(0.5, np.sqrt(0.5))))
    assert az[3] == pytest.approx(30.0)


def test_compact_array_masks_nonphysical_and_endfire_bins_instead_of_rejecting_grid():
    cube = np.arange(20.0).reshape(4, 5, 1, 1)
    beams, az, el = visible_array_beams(
        cube, np.array([2.0, 1.0, 0.0, -1.0]), np.array([-2.0, -1.0, 0.0, 0.5, 1.0])
    )
    np.testing.assert_array_equal(beams[:, 0, 0], [12.0, 13.0])
    np.testing.assert_allclose(az, [0.0, 30.0])
    np.testing.assert_array_equal(el, [0.0, 0.0])


@pytest.mark.parametrize("bad", [np.nan, -1.0, np.inf])
def test_nonphysical_power_is_rejected(bad):
    beams, rd, cfg = _inputs()
    beams[0, 1, 1] = bad
    with pytest.raises(ValueError):
        run_beam_scan_detection_pipeline(
            rd,
            beam_power=beams,
            beam_azimuth_deg=np.zeros(3),
            beam_elevation_deg=np.zeros(3),
            config=cfg,
            wavelength_m=1.0,
        )


@pytest.mark.parametrize(
    "rows,cols,y,z", [(1, 32, 0.37, 0.0), (8, 32, -0.37, -0.18), (8, 64, 0.41, 0.23)]
)
def test_refined_angle_matches_independent_off_bin_wave_without_changing_detection(
    rows, cols, y, z
):
    from copy import deepcopy

    from shared.sensing.arrays import planar_spatial_fft
    from shared.sensing.detection.beam_scan import refine_array_directions

    coordinates = np.array([(c * 0.5, -r * 0.5) for c in range(cols) for r in range(rows)])
    signal = np.exp(2j * np.pi * (coordinates[:, 0] * y + coordinates[:, 1] * z))[:, None, None]
    spatial = planar_spatial_fft(
        signal,
        num_rows=rows,
        num_cols=cols,
        horizontal_spacing_wavelengths=0.5,
        vertical_spacing_wavelengths=0.5,
    )
    beams, az, el = visible_array_beams(
        abs(spatial.spectrum) ** 2,
        spatial.elevation_direction_cosine,
        spatial.azimuth_direction_cosine,
    )
    k = int(beams[:, 0, 0].argmax())
    detection = dict(
        range_m=30.0,
        velocity_m_s=1.5,
        range_bin=0,
        doppler_bin=0,
        aoa_az_deg=az[k],
        aoa_el_deg=el[k],
        beam_index=k,
        power_linear=123.0,
    )
    before = deepcopy(detection)
    result = refine_array_directions(
        [detection],
        signal,
        num_rows=rows,
        num_cols=cols,
        horizontal_spacing_wavelengths=0.5,
        vertical_spacing_wavelengths=0.5,
        oversampling=8,
    )[0]
    assert detection == before
    assert result["beam_index"] == k and result["power_linear"] == 123.0
    assert result["range_m"] == 30.0 and result["velocity_m_s"] == 1.5
    assert result["coarse_aoa_az_deg"] == az[k]
    truth = 30 * np.array([np.sqrt(1 - y * y - z * z), y, z])
    refined = np.array([result["x_m"], result["y_m"], result["z_m"]])
    assert np.linalg.norm(refined - truth) < 0.5
    assert result["beam_angle_oversampling"] == 8
    if rows == 1:
        assert result["aoa_el_deg"] == 0.0


@pytest.mark.parametrize("factor", [0, 17, True, 1.5])
def test_angle_refinement_rejects_invalid_factor(factor):
    from shared.sensing.detection.beam_scan import refine_array_directions

    with pytest.raises(ValueError, match="integer from 1 to 16"):
        refine_array_directions(
            [],
            np.ones((4, 1, 1), complex),
            num_rows=1,
            num_cols=4,
            horizontal_spacing_wavelengths=0.5,
            vertical_spacing_wavelengths=0.5,
            oversampling=factor,
        )


def test_angle_refinement_does_not_invent_direction_from_power_only_or_empty_signal():
    from shared.sensing.detection.beam_scan import refine_array_directions

    kw = dict(
        num_rows=1,
        num_cols=4,
        horizontal_spacing_wavelengths=0.5,
        vertical_spacing_wavelengths=0.5,
        oversampling=8,
    )
    with pytest.raises(ValueError, match="finite complex"):
        refine_array_directions([], np.ones((4, 1, 1)), **kw)
    d = dict(
        range_m=10.0, velocity_m_s=1.0, range_bin=0, doppler_bin=0, aoa_az_deg=0.0, aoa_el_deg=0.0
    )
    with pytest.raises(ValueError, match="signal energy"):
        refine_array_directions([d], np.zeros((4, 1, 1), complex), **kw)
    with pytest.raises(ValueError, match="spacings"):
        refine_array_directions(
            [], np.ones((4, 1, 1), complex), **{**kw, "horizontal_spacing_wavelengths": 0.6}
        )
