from __future__ import annotations

import numpy as np
import pytest

from shared.sensing.detection.pipeline import (
    DetectionPipelineConfig,
    run_vote_fusion_pipeline,
)
from shared.sensing.detection.vote import ProjectionCfarConfig, VoteFusionConfig, vote_fusion_3d
from shared.sensing.signal_processing import RangeDopplerResult


def _make_vote_config(
    *,
    min_votes: int = 2,
    edge_policy: str = "full_window",
    pre_threshold_db: float = 20.0,
) -> VoteFusionConfig:
    projection = ProjectionCfarConfig(
        guard_cells=(1, 1),
        reference_cells=(1, 1),
        threshold_scale=1.75,
        edge_policy=edge_policy,
    )
    return VoteFusionConfig(
        rd=projection,
        raz=projection,
        daz=projection,
        min_votes=min_votes,
        pre_threshold_db=pre_threshold_db,
        nms_suppression_radius=(1, 1, 1),
        cluster_range_gate_m=1.0,
        cluster_velocity_gate_m_s=1.0,
    )


def _make_rd_result_from_cube(cube: np.ndarray) -> RangeDopplerResult:
    doppler_bins = cube.shape[1]
    range_bins = cube.shape[2]
    return RangeDopplerResult(
        power_linear=cube.sum(axis=0),
        range_axis_m=np.arange(range_bins, dtype=np.float64),
        doppler_axis_hz=np.linspace(-4.0, 3.0, doppler_bins, dtype=np.float64),
        cube=None,
    )


def test_vote_fusion_keeps_strong_target_when_projection_peaks_agree() -> None:
    cube = np.ones((6, 8, 12), dtype=np.float64)
    cube[2, 5, 7] += 100.0
    cube[:, 2, 9] += 4.0
    azimuth_axis_deg = np.linspace(-45.0, 45.0, cube.shape[0], dtype=np.float64)

    result = vote_fusion_3d(
        cube,
        azimuth_axis_deg=azimuth_axis_deg,
        config=_make_vote_config(min_votes=2),
    )

    assert result.votes[2, 5, 7] == 3
    assert result.fused_mask[2, 5, 7]
    assert result.rd_peak_mask[5, 7]
    assert np.isfinite(result.aoa_az_map[5, 7])


def test_vote_fusion_min_votes_controls_two_of_three_consistency() -> None:
    cube = np.ones((6, 8, 12), dtype=np.float64)
    cube[:, 3, 6] += 4.0
    cube[4, :, 6] += 4.0
    azimuth_axis_deg = np.linspace(-45.0, 45.0, cube.shape[0], dtype=np.float64)

    result_two_votes = vote_fusion_3d(
        cube,
        azimuth_axis_deg=azimuth_axis_deg,
        config=_make_vote_config(min_votes=2, edge_policy="truncate"),
    )
    result_three_votes = vote_fusion_3d(
        cube,
        azimuth_axis_deg=azimuth_axis_deg,
        config=_make_vote_config(min_votes=3, edge_policy="truncate"),
    )

    assert result_two_votes.votes[4, 3, 6] == 2
    assert result_two_votes.fused_mask[4, 3, 6]
    assert not result_three_votes.fused_mask[4, 3, 6]


def test_vote_fusion_exposes_edge_policy_and_exact_edge_training_counts() -> None:
    cube = np.ones((6, 8, 12), dtype=np.float64)
    cube[0, 0, 0] = 100.0
    azimuth_axis_deg = np.linspace(-45.0, 45.0, cube.shape[0], dtype=np.float64)

    full = vote_fusion_3d(
        cube,
        azimuth_axis_deg=azimuth_axis_deg,
        config=_make_vote_config(min_votes=1),
    )
    truncated = vote_fusion_3d(
        cube,
        azimuth_axis_deg=azimuth_axis_deg,
        config=_make_vote_config(min_votes=1, edge_policy="truncate"),
    )

    assert full.projection_edge_policies == {
        "rd": "full_window",
        "raz": "full_window",
        "daz": "full_window",
    }
    assert truncated.projection_edge_policies == {
        "rd": "truncate",
        "raz": "truncate",
        "daz": "truncate",
    }
    assert not full.rd_cfar.eligible[0, 0]
    assert full.rd_cfar.training_cell_count[0, 0] == 0
    assert full.votes[0, 0, 0] == 0
    assert truncated.rd_cfar.eligible[0, 0]
    # Clipped 3x3 outer window minus the clipped 2x2 guard-plus-CUT block.
    assert truncated.rd_cfar.training_cell_count[0, 0] == 5
    assert truncated.votes[0, 0, 0] == 3


def test_vote_fusion_rejects_unknown_projection_edge_policy() -> None:
    with pytest.raises(ValueError, match="edge_policy"):
        vote_fusion_3d(
            np.ones((6, 8, 12), dtype=np.float64),
            azimuth_axis_deg=np.linspace(-45.0, 45.0, 6, dtype=np.float64),
            config=_make_vote_config(edge_policy="wrap"),
        )


def test_run_vote_fusion_pipeline_emits_true_3d_detection_fields() -> None:
    cube = np.ones((7, 8, 12), dtype=np.float64)
    cube[3, 5, 7] += 100.0
    azimuth_axis_deg = np.linspace(-60.0, 60.0, cube.shape[0], dtype=np.float64)
    rd_result = _make_rd_result_from_cube(cube)

    result = run_vote_fusion_pipeline(
        rd_result,
        azimuth_cube=cube,
        azimuth_axis_deg=azimuth_axis_deg,
        config=DetectionPipelineConfig(
            cfar_guard_cells=(1, 1),
            cfar_reference_cells=(1, 1),
            # Deliberately differs from the vote RD projection. Metadata must
            # describe the detector that actually ran, not this unused value.
            cfar_threshold_scale=9.0,
            nms_suppression_radius=(1, 1),
            cluster_range_gate_m=1.0,
            cluster_velocity_gate_m_s=1.0,
            min_snr_db=0.0,
        ),
        # Preserve a positive background so the pipeline's explicit SNR
        # denominator-validity contract can evaluate the detected cell.
        vote_config=_make_vote_config(min_votes=2, pre_threshold_db=0.0),
        wavelength_m=1.0,
        return_stage_artifacts=True,
    )

    assert result.metadata["detection_mode"] == "vote_fusion_3d"
    assert result.metadata["cfar"]["threshold_scale"] == pytest.approx(1.75)
    assert result.metadata["cfar"]["os_rank_rounding"].endswith(
        "ties_to_even_of_fraction_times_k_minus_one"
    )
    assert len(result.detections) == 1
    det = result.detections[0]
    assert det["azimuth_bin"] == 3
    assert det["vote_count"] == 3
    assert det["aoa_az_deg"] == pytest.approx(azimuth_axis_deg[3])
    assert det["path_closing_rate_m_s"] == pytest.approx(det["doppler_hz"])
    assert det["monostatic_equivalent_velocity_m_s"] == pytest.approx(
        det["path_closing_rate_m_s"] / 2.0
    )
    assert det["velocity_m_s"] == pytest.approx(det["monostatic_equivalent_velocity_m_s"])
    assert result.stages is not None
    assert result.stages.extra_artifacts is not None
    assert result.stages.extra_artifacts["vote_counts"].shape == cube.shape
    assert result.stages.extra_artifacts["vote_cluster_labels"].shape == cube.shape
