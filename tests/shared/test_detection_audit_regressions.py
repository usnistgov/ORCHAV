"""Independent physical-reference regressions for the release audit findings."""

import builtins

import numpy as np
import pytest

from shared.sensing.detection.angle import AnglePeakDetectionConfig
from shared.sensing.detection.cfar import cfar_1d, cfar_2d, nms_peaks_2d
from shared.sensing.detection.pipeline import (
    DetectionPipelineConfig,
    run_angle_enhanced_detection_pipeline,
    run_detection_pipeline,
    run_vote_fusion_pipeline,
)
from shared.sensing.detection.vote import (
    ProjectionCfarConfig,
    VoteFusionConfig,
    cluster_detections_3d,
)
from shared.sensing.signal_processing import RangeDopplerResult


def rd_result(power):
    return RangeDopplerResult(
        power, np.arange(float(power.shape[1])), np.arange(float(power.shape[0])), None
    )


def test_nms_must_not_compare_opposite_range_edges():
    power = np.zeros((5, 5))
    power[2, 0], power[2, 4] = 5.0, 10.0
    result = nms_peaks_2d(power > 0, power, suppression_radius=(1, 1))
    assert np.argwhere(result).tolist() == [[2, 0], [2, 4]]


def test_truncated_pipeline_retains_independent_range_edge_targets():
    power = np.ones((9, 9))
    power[4, 0], power[4, 8] = 100.0, 200.0
    config = DetectionPipelineConfig(
        (1, 1),
        (1, 1),
        2,
        cfar_edge_policy="truncate",
        nms_suppression_radius=(1, 1),
        cluster_range_gate_m=0.1,
        min_snr_db=6,
    )
    result = run_detection_pipeline(
        rd_result(power), config=config, wavelength_m=1, return_stage_artifacts=True
    )
    assert np.argwhere(result.stages.cfar_mask).tolist() == [[4, 0], [4, 8]]
    assert {d["range_bin"] for d in result.detections} == {0, 8}


def test_3d_cluster_snr_comes_from_detected_peak_not_empty_centroid():
    mask = np.zeros((1, 3, 5), bool)
    mask[0, 1, [1, 3]] = True
    power = np.zeros(mask.shape)
    power[mask] = [10.0, 6.0]
    snr = np.full((3, 5), -300.0)
    snr[1, [1, 3]] = [12.0, 7.0]
    result = cluster_detections_3d(
        mask,
        power,
        np.arange(5.0),
        np.arange(3.0),
        [0],
        wavelength=1,
        snr_map=snr,
        cluster_range_gate_m=3,
    )
    assert len(result) == 1
    assert result[0].snr_db == 12.0
    assert result[0].range_bin == 1
    assert result[0].centroid_range_m == pytest.approx(1.75, abs=1e-14)


def test_angle_enhancement_does_not_drop_high_snr_cluster_in_valley():
    power = np.ones((9, 9))
    power[4, [3, 5]] = 100.0
    cube = np.ones((3, 9, 9))
    cube[1, 4, [3, 5]] = 100.0
    config = DetectionPipelineConfig(
        (0, 0), (1, 1), 2, nms_suppression_radius=(0, 0), cluster_range_gate_m=3, min_snr_db=6
    )
    base = run_detection_pipeline(rd_result(power), config=config, wavelength_m=1)
    angle = run_angle_enhanced_detection_pipeline(
        rd_result(power),
        config=config,
        azimuth_cube=cube,
        azimuth_axis_deg=[-10, 0, 10],
        angle_config=AnglePeakDetectionConfig(max_peaks_per_rd_cell=1),
        wavelength_m=1,
        return_stage_artifacts=True,
    )
    assert len(base.detections) == 1
    assert angle.metadata["clusters"] == 1
    assert len(angle.detections) == 1


def vote_pipeline(peak, pre_threshold_db=20):
    cube = np.ones((7, 9, 13))
    cube[3, 4, 6] = peak
    projection = ProjectionCfarConfig((1, 1), (1, 1), 1.75)
    vote = VoteFusionConfig(projection, projection, projection, pre_threshold_db=pre_threshold_db)
    config = DetectionPipelineConfig((1, 1), (1, 1), 1.75, min_snr_db=6)
    return run_vote_fusion_pipeline(
        rd_result(cube.sum(axis=0)),
        azimuth_cube=cube,
        azimuth_axis_deg=np.linspace(-60, 60, 7),
        config=config,
        vote_config=vote,
        wavelength_m=1,
        return_stage_artifacts=True,
    )


def test_vote_default_prethreshold_must_not_invalidate_measured_noise():
    weaker, stronger, control = vote_pipeline(100), vote_pipeline(101), vote_pipeline(101, 0)
    assert len(weaker.detections) == len(control.detections) == 1
    assert stronger.metadata["raw_fused_cells"] == 1
    assert len(stronger.detections) == 1


def test_3d_apis_reject_unsupported_dynamic_range_floor():
    power = np.ones((13, 17))
    power[4, 4], power[8, 12] = 1000.0, 50.0
    cube = np.stack([0.1 * power, 0.8 * power, 0.1 * power])
    config = DetectionPipelineConfig((1, 1), (1, 1), 2, min_snr_db=6, cfar_max_dynamic_range_db=10)
    base = run_detection_pipeline(rd_result(power), config=config, wavelength_m=1)
    assert {d["range_bin"] for d in base.detections} == {4}
    with pytest.raises(ValueError, match="cfar_max_dynamic_range_db supports only"):
        run_angle_enhanced_detection_pipeline(
            rd_result(power),
            azimuth_cube=cube,
            azimuth_axis_deg=[-10, 0, 10],
            config=config,
            angle_config=AnglePeakDetectionConfig(max_peaks_per_rd_cell=1),
            wavelength_m=1,
            return_stage_artifacts=True,
        )
    projection = ProjectionCfarConfig((1, 1), (1, 1), 2)
    with pytest.raises(ValueError, match="cfar_max_dynamic_range_db supports only"):
        run_vote_fusion_pipeline(
            rd_result(power),
            azimuth_cube=cube,
            azimuth_axis_deg=[-10, 0, 10],
            config=config,
            vote_config=VoteFusionConfig(projection, projection, projection),
            wavelength_m=1,
        )


def test_ca_1d_distant_strong_cell_does_not_erase_training_noise():
    power = np.ones(9)
    power[0] = 1e16
    result = cfar_1d(power, guard_cells=0, reference_cells=1, threshold_scale=2)
    np.testing.assert_allclose(result.noise[2:-1], 1.0)
    assert not result.detections.any()


def test_cluster_3d_optional_dependency_does_not_change_cluster_semantics(monkeypatch):
    power = np.ones((1, 1, 2))
    args = (power > 0, power, np.array([0.0, 0.5]), np.array([0.0]), np.array([0.0]))
    with_sklearn = cluster_detections_3d(*args, wavelength=1)
    original_import = builtins.__import__

    def import_without_sklearn(name, *args, **kwargs):
        if name == "sklearn.cluster":
            raise ImportError("Simulated optional dependency absence")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_sklearn)
    without_sklearn = cluster_detections_3d(*args, wavelength=1)
    assert len(with_sklearn) == 1
    assert len(without_sklearn) == 1


@pytest.mark.parametrize("method", ["ca", "os"])
@pytest.mark.parametrize("edge", ["full_window", "truncate"])
def test_cfar_2d_matches_independent_direct_reference_at_ordinary_scale(method, edge):
    rng = np.random.default_rng(92741)
    power = rng.exponential(size=(17, 23))
    gd, gr, rd, rr = 1, 2, 2, 3
    result = cfar_2d(
        power,
        guard_cells=(gd, gr),
        reference_cells=(rd, rr),
        threshold_scale=3,
        method=method,
        os_rank=0.75,
        edge_policy=edge,
    )
    expected_noise = np.zeros_like(power)
    expected_count = np.zeros_like(power, dtype=int)
    for i in range(power.shape[0]):
        for j in range(power.shape[1]):
            if edge == "full_window" and (
                i < gd + rd
                or i + gd + rd >= power.shape[0]
                or j < gr + rr
                or j + gr + rr >= power.shape[1]
            ):
                continue
            samples = [
                power[k, column]
                for k in range(max(0, i - gd - rd), min(power.shape[0], i + gd + rd + 1))
                for column in range(max(0, j - gr - rr), min(power.shape[1], j + gr + rr + 1))
                if abs(k - i) > gd or abs(column - j) > gr
            ]
            expected_count[i, j] = len(samples)
            expected_noise[i, j] = (
                sum(samples) / len(samples)
                if method == "ca"
                else sorted(samples)[round(0.75 * (len(samples) - 1))]
            )
    np.testing.assert_array_equal(result.training_cell_count, expected_count)
    np.testing.assert_allclose(result.noise, expected_noise, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(
        result.detections, (power > 3 * expected_noise) & (expected_count > 0)
    )


@pytest.mark.parametrize(
    "peaks", [((0, 2), (4, 2)), ((0, 0), (4, 4)), ((0, 4), (4, 0)), ((2, 0), (2, 4))]
)
def test_nms_all_opposite_edges_and_corners_are_independent(peaks):
    power = np.zeros((5, 5))
    power[peaks[0]], power[peaks[1]] = 5.0, 10.0
    result = nms_peaks_2d(power > 0, power, suppression_radius=(1, 1))
    assert {tuple(i) for i in np.argwhere(result)} == set(peaks)


@pytest.mark.parametrize(
    "peaks", [((0, 0), (1, 0)), ((4, 4), (3, 4)), ((0, 4), (0, 3)), ((4, 0), (4, 1))]
)
def test_nms_adjacent_boundary_neighbours_are_compared(peaks):
    power = np.zeros((5, 5))
    power[peaks[0]], power[peaks[1]] = 5.0, 10.0
    assert np.argwhere(nms_peaks_2d(power > 0, power)).tolist() == [list(peaks[1])]


@pytest.mark.parametrize("peak", [100.0, 101.0, 1000.0, 1e6])
def test_vote_amplitude_sweep_preserves_uncensored_projection_noise(peak):
    result = vote_pipeline(peak)
    assert len(result.detections) == 1
    assert result.stages.cfar_noise_floor[4, 6] == pytest.approx(7.0)


@pytest.mark.parametrize("location", [(0, 0), (3, 7), (8, 10), (12, 14)])
@pytest.mark.parametrize("edge", ["full_window", "truncate"])
def test_extreme_dynamic_range_cfar_matches_direct_annulus(location, edge):
    import math

    power = np.ones((13, 15))
    power[location] = 1e18
    result = cfar_2d(
        power, guard_cells=(1, 1), reference_cells=(2, 2), threshold_scale=2, edge_policy=edge
    )
    for i, j in np.argwhere(result.eligible):
        training = [
            power[k, column]
            for k in range(max(0, i - 3), min(13, i + 4))
            for column in range(max(0, j - 3), min(15, j + 4))
            if abs(k - i) > 1 or abs(column - j) > 1
        ]
        expected = math.fsum(training) / len(training)
        assert result.noise[i, j] == pytest.approx(expected, rel=1e-12)
        assert result.detections[i, j] == (power[i, j] > 2 * expected)


def test_3d_snr_validity_removes_invalid_candidate_before_clustering():
    power = np.ones((1, 3, 5))
    mask = np.zeros_like(power, bool)
    mask[0, 1, [1, 3]] = True
    power[0, 1, [1, 3]] = [100.0, 10.0]
    valid = np.ones((3, 5), bool)
    valid[1, 1] = False
    clusters = cluster_detections_3d(
        mask,
        power,
        np.arange(5.0),
        np.arange(3.0),
        [0.0],
        wavelength=1,
        snr_map=np.ones((3, 5)) * 12,
        snr_valid_mask=valid,
        cluster_range_gate_m=3,
    )
    assert len(clusters) == 1
    assert clusters[0].range_bin == 3
    assert clusters[0].num_detections == 1


def test_sparse_zero_training_rings_do_not_require_dense_fallback(monkeypatch):
    import shared.sensing.detection.cfar as cfar

    power = np.zeros((64, 64))
    power[32, 32] = 100.0

    def reject_unnecessary_local_sum(_values):
        raise AssertionError("A genuinely zero training ring needs no local fallback")

    monkeypatch.setattr(cfar.math, "fsum", reject_unnecessary_local_sum)
    result = cfar_2d(power, guard_cells=(1, 1), reference_cells=(2, 2), threshold_scale=2)
    assert result.noise[32, 32] == 0.0
    assert result.detections[32, 32]


@pytest.mark.parametrize("dynamic_range_db", [80, 140, 160, 170, 300])
def test_pipeline_retains_weak_target_with_extreme_background_ratio(dynamic_range_db):
    background = 10.0 ** (-dynamic_range_db / 10.0)
    power = np.full((64, 64), background)
    power[10, 10], power[45, 45] = 1.0, 1e-4
    result = run_detection_pipeline(
        rd_result(power),
        wavelength_m=1.0,
        config=DetectionPipelineConfig(
            (1, 1),
            (3, 3),
            3.0,
            cfar_max_dynamic_range_db=60.0,
            cluster_range_gate_m=0.1,
            min_snr_db=6,
        ),
        return_stage_artifacts=True,
    )
    assert {d["range_bin"] for d in result.detections} == {10, 45}
    assert result.stages.cfar_noise_floor[45, 45] == pytest.approx(background, rel=1e-12, abs=0.0)
