"""Regression controls for eligibility, physical centroids, and 3-D contracts."""

from dataclasses import replace

import numpy as np
import pytest

from shared.sensing.detection.pipeline import (
    DetectionPipelineConfig,
    apply_ghost_filter,
    run_detection_pipeline,
    run_vote_fusion_pipeline,
)
from shared.sensing.detection.vote import (
    ProjectionCfarConfig,
    VoteFusionConfig,
    cluster_detections_3d,
    nms_peaks_3d,
    vote_fusion_3d,
)
from shared.sensing.signal_processing import RangeDopplerResult


def _projection(refs=(1, 1)):
    return ProjectionCfarConfig((0, 0), refs, 2)


def _vote_case(*, wrapper=False, radius=(1, 2, 4), quota=200):
    cube = np.ones((9, 9, 9))
    cube[4, 1, 4] = 1000
    cube[4, 3, 4] = 500
    axis = np.arange(9.0)
    projection = _projection((1, 2) if wrapper else (1, 1))
    config = VoteFusionConfig(
        _projection((2, 2)),
        projection,
        projection,
        min_votes=1 if wrapper else 2,
        nms_suppression_radius=radius,
        nms_max_peaks=quota,
    )
    rd = RangeDopplerResult(cube.sum(axis=0), axis, axis, None)
    return run_vote_fusion_pipeline(
        rd,
        azimuth_cube=cube,
        azimuth_axis_deg=axis,
        config=DetectionPipelineConfig((0, 0), (1, 1), 2),
        vote_config=config,
        wavelength_m=1,
        return_stage_artifacts=True,
    )


@pytest.mark.parametrize("wrapper", [False, True])
@pytest.mark.parametrize("radius,quota", [((1, 2, 4), 200), ((0, 0, 0), 1)])
def test_ineligible_vote_candidate_cannot_suppress_or_spend_quota(wrapper, radius, quota):
    result = _vote_case(wrapper=wrapper, radius=radius, quota=quota)
    assert len(result.detections) == 1
    assert result.detections[0]["doppler_bin"] == 3
    assert not result.stages.snr_valid_mask[1, 4]
    assert result.stages.snr_valid_mask[3, 4]
    assert result.stages.snr_map[3, 4] > 6


@pytest.mark.parametrize("ranges,doppler", [([0, 1, 10], [0, 1, 10]), ([0, 5, 10], [-5, 0, 5])])
def test_centroid_is_power_weighted_physical_mean(ranges, doppler):
    cube = np.zeros((1, 3, 3))
    cube[0, 0, 0], cube[0, 2, 2] = 1, 3
    det = cluster_detections_3d(
        cube > 0,
        cube,
        ranges,
        doppler,
        [0],
        wavelength=2,
        cluster_range_gate_m=100,
        cluster_velocity_gate_m_s=100,
    )[0]
    assert det.centroid_range_m == pytest.approx(np.average([ranges[0], ranges[2]], weights=[1, 3]))
    assert det.centroid_doppler_hz == pytest.approx(
        np.average([doppler[0], doppler[2]], weights=[1, 3])
    )
    assert (det.range_bin, det.doppler_bin) == (2, 2)
    assert det.range_m == ranges[2]


@pytest.mark.parametrize(
    "angles,gate,count",
    [([-179, 179], 5, 1), ([-179, 179], 1, 2), ([-10, 10], 5, 2), ([-1, 1], 5, 1)],
)
def test_azimuth_neighbours_use_shortest_circular_distance(angles, gate, count):
    cube = np.ones((2, 1, 1))
    clusters = cluster_detections_3d(
        cube > 0, cube, [1], [0], angles, wavelength=1, cluster_azimuth_gate_deg=gate
    )
    assert len(clusters) == count
    if count == 1 and angles[0] == -179:
        assert abs(clusters[0].centroid_aoa_az_deg) == pytest.approx(180)


def test_seam_uses_ellipsoidal_arc_distance_and_doppler_does_not_wrap():
    cube = np.zeros((2, 2, 1))
    cube[0, 0, 0] = cube[1, 1, 0] = 1
    # Each coordinate is individually inside its gate, but sum of squares > 1.
    clusters = cluster_detections_3d(
        cube > 0,
        cube,
        [1],
        [0, 8],
        [-179, 179],
        wavelength=2,
        cluster_velocity_gate_m_s=10,
        cluster_azimuth_gate_deg=3,
    )
    assert len(clusters) == 2
    clusters = cluster_detections_3d(
        cube > 0,
        cube,
        [1],
        [-100, 100],
        [-179, 179],
        wavelength=2,
        cluster_velocity_gate_m_s=10,
        cluster_azimuth_gate_deg=5,
    )
    assert len(clusters) == 2


@pytest.mark.parametrize(
    "change",
    [
        dict(guard_cells=(0.5, 0)),
        dict(reference_cells=(1, 1.5)),
        dict(method="typo"),
        dict(method="CA"),
    ],
)
def test_standalone_vote_rejects_lossy_projection_config(change):
    proj = replace(_projection(), **change)
    with pytest.raises(ValueError):
        vote_fusion_3d(
            np.ones((3, 3, 3)),
            azimuth_axis_deg=[0, 1, 2],
            config=VoteFusionConfig(proj, proj, proj),
        )


@pytest.mark.parametrize(
    "change",
    [
        dict(min_votes=1.5),
        dict(min_votes=True),
        dict(pre_threshold_db=np.nan),
        dict(nms_max_peaks=0),
        dict(post_nms_min_peak_divisor=0.5),
        dict(cluster_azimuth_gate_deg=-1),
    ],
)
def test_standalone_vote_validates_all_configuration(change):
    proj = _projection()
    with pytest.raises(ValueError):
        vote_fusion_3d(
            np.ones((3, 3, 3)),
            azimuth_axis_deg=[0, 1, 2],
            config=replace(VoteFusionConfig(proj, proj, proj), **change),
        )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1, 1j])
def test_nms_rejects_invalid_power_even_when_mask_empty(bad):
    cube = np.full((1, 1, 1), bad)
    with pytest.raises(ValueError):
        nms_peaks_3d(np.zeros(cube.shape, bool), cube)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(max_peaks=0),
        dict(max_peaks=-1),
        dict(max_peaks=1.5),
        dict(suppression_radius=(-1, 0, 0)),
        dict(suppression_radius=(0.5, 0, 0)),
        dict(suppression_radius=(0, 0)),
    ],
)
def test_nms_rejects_bad_counts_even_when_empty(kwargs):
    with pytest.raises(ValueError):
        nms_peaks_3d(np.zeros((0, 1, 1), bool), np.zeros((0, 1, 1)), **kwargs)


def test_nms_stable_row_major_ties_zero_radius_and_empty_cube():
    cube = np.ones((2, 3, 4))
    mask = cube > 0
    kept = nms_peaks_3d(mask, cube, suppression_radius=(0, 0, 0), max_peaks=10)
    assert np.flatnonzero(kept).tolist() == list(range(10))
    assert nms_peaks_3d(mask, cube, suppression_radius=(2, 3, 4)).sum() == 1
    assert nms_peaks_3d(np.zeros((0, 1, 1), bool), np.zeros((0, 1, 1))).shape == (0, 1, 1)


@pytest.mark.parametrize(
    "change",
    [
        dict(wavelength=0),
        dict(cluster_range_gate_m=0),
        dict(cluster_velocity_gate_m_s=np.inf),
        dict(cluster_azimuth_gate_deg=-1),
        dict(range_mode="bad"),
    ],
)
def test_standalone_clustering_rejects_invalid_physical_contract(change):
    kwargs = dict(wavelength=1)
    kwargs.update(change)
    with pytest.raises(ValueError):
        cluster_detections_3d(np.ones((1, 1, 1), bool), np.ones((1, 1, 1)), [1], [0], [0], **kwargs)


def test_standalone_clustering_rejects_zero_power_selection():
    with pytest.raises(ValueError, match="positive power"):
        cluster_detections_3d(
            np.ones((1, 1, 1), bool), np.zeros((1, 1, 1)), [1], [0], [0], wavelength=1
        )


def test_noiseless_rejection_and_ghost_heuristic_remain_explicit_policies():
    power = np.zeros((9, 9))
    power[4, 4] = 100
    result = run_detection_pipeline(
        RangeDopplerResult(power, np.arange(9.0), np.arange(9.0), None),
        config=DetectionPipelineConfig((0, 0), (1, 1), 2),
        wavelength_m=1,
        return_stage_artifacts=True,
    )
    assert result.stages.cfar_mask[4, 4]
    assert not result.stages.snr_valid_mask[4, 4]
    assert not result.detections
    dets = [
        dict(range_m=2.0, velocity_m_s=0.0, aoa_az_deg=0.0),
        dict(range_m=8.0, velocity_m_s=0.0, aoa_az_deg=80.0),
    ]
    assert len(apply_ghost_filter(dets, velocity_gate_m_s=1)[0]) == 1
    assert len(apply_ghost_filter(dets, velocity_gate_m_s=1, azimuth_gate_deg=10)[0]) == 2
    dets[1]["aoa_az_deg"] = np.nan
    assert len(apply_ghost_filter(dets, velocity_gate_m_s=1, azimuth_gate_deg=10)[0]) == 1


def test_large_finite_prethreshold_does_not_overflow():
    proj = _projection()
    result = vote_fusion_3d(
        np.ones((3, 3, 3)),
        azimuth_axis_deg=[0, 1, 2],
        config=VoteFusionConfig(proj, proj, proj, pre_threshold_db=1e308),
    )
    assert np.all(np.isfinite(result.power_cube))
