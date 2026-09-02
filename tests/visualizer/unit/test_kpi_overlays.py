"""Tests for per-frame KPI overlay features.

Validates:
- Per-frame direct-path pair share from represented pair topology
- Per-frame path count and strongest single-path loss extraction
- Frame-index-to-metric mapping for trajectory coloring
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from shared.frames import FrameProjection, ProjectedMPCFrame
from shared.frames.contracts import PATH_METRIC_VALIDITY_BITS
from visualizer.src.metrics.scenario_statistics import (
    SCENARIO_STATISTICS_REQUEST,
    ScenarioStatisticsAccumulator,
)
from visualizer.src.services.trajectory_load_service import TrajectoryLoadCoordinator


def _make_frame(
    *,
    n_paths: int = 10,
    interactions: np.ndarray | None = None,
    path_loss_db: np.ndarray | None = None,
    delays_ns: np.ndarray | None = None,
    frame_index: int = 0,
) -> FrameProjection:
    """Build the selective projection required by scenario statistics.

    Args:
        n_paths: Number of paths to generate.
        interactions: Per-path physical interaction codes, zero padded by row.
        path_loss_db: Custom per-path loss array.
        delays_ns: Custom per-path delay array.
        frame_index: Scenario timeline index.

    Returns:
        A projection containing topology, interactions, and path metrics.
    """
    if interactions is None:
        interactions = np.zeros((n_paths, 3), dtype=np.uint8)
        interactions[:, 0] = 1
    if path_loss_db is None:
        path_loss_db = np.linspace(50.0, 90.0, n_paths, dtype=np.float32)
    if delays_ns is None:
        delays_ns = np.linspace(10.0, 100.0, n_paths, dtype=np.float32)

    interaction_rows = np.asarray(interactions, dtype=np.uint8)
    bounce_counts = np.count_nonzero(interaction_rows, axis=1)
    bounce_offsets = np.zeros((n_paths + 1,), dtype=np.int64)
    bounce_offsets[1:] = np.cumsum(bounce_counts)
    metric_valid_bits = np.full(
        (n_paths,),
        np.uint8(sum(PATH_METRIC_VALIDITY_BITS.values())),
        dtype=np.uint8,
    )
    angles = np.zeros((n_paths,), dtype=np.float32)
    frame = ProjectedMPCFrame(
        frame_index=frame_index,
        tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int32),
        pair_path_offsets=np.asarray([0, n_paths], dtype=np.int64),
        bounce_offsets=bounce_offsets,
        interactions=interaction_rows[interaction_rows != 0],
        delays_ns=np.asarray(delays_ns, dtype=np.float32),
        path_loss_db=np.asarray(path_loss_db, dtype=np.float32),
        aoa_az_deg=angles,
        aoa_el_deg=angles,
        aod_az_deg=angles,
        aod_el_deg=angles,
        metric_valid_bits=metric_valid_bits,
    )
    return FrameProjection.from_request(frame, SCENARIO_STATISTICS_REQUEST)


def _make_direct_pair_frame(n_paths: int = 10, *, frame_index: int = 0) -> FrameProjection:
    """Build one represented pair with at least one zero-bounce direct path.

    The remaining paths have one specular interaction.
    """
    interactions = np.zeros((n_paths, 3), dtype=np.uint8)
    interactions[1:, 0] = 1
    return _make_frame(
        n_paths=n_paths,
        interactions=interactions,
        frame_index=frame_index,
    )


def _make_indirect_pair_frame(n_paths: int = 10, *, frame_index: int = 0) -> FrameProjection:
    """Build one represented pair whose paths all have an interaction."""
    interactions = np.zeros((n_paths, 3), dtype=np.uint8)
    interactions[:, 0] = 1
    return _make_frame(
        n_paths=n_paths,
        interactions=interactions,
        frame_index=frame_index,
    )


class TestPerFrameKPIExtraction:
    """Test that projection aggregation computes per-frame KPIs."""

    def _collect(self, projections: list[FrameProjection]) -> dict[str, Any]:
        """Aggregate a complete sequence of statistics projections."""
        return ScenarioStatisticsAccumulator().collect_from_projections(projections)

    def test_direct_path_pair_share_is_one_for_single_direct_pair(self) -> None:
        frame = _make_direct_pair_frame(n_paths=5)
        stats = self._collect([frame])

        assert stats["direct_path_pair_share_evolution"] == [1.0]
        assert stats["pair_visibility_evolution"]["direct_path_present"] == [1]

    def test_direct_path_pair_share_is_zero_for_single_indirect_pair(self) -> None:
        frame = _make_indirect_pair_frame(n_paths=5)
        stats = self._collect([frame])

        assert stats["direct_path_pair_share_evolution"] == [0.0]
        assert stats["pair_visibility_evolution"]["indirect_only"] == [1]

    def test_direct_path_pair_share_tracks_direct_indirect_and_empty_frames(self) -> None:
        stats = self._collect(
            [
                _make_direct_pair_frame(n_paths=5, frame_index=0),
                _make_indirect_pair_frame(n_paths=5, frame_index=1),
                _make_frame(n_paths=0, frame_index=2),
            ]
        )

        assert stats["direct_path_pair_share_evolution"] == [1.0, 0.0, 0.0]
        assert stats["pair_visibility_evolution"] == {
            "direct_path_present": [1, 0, 0],
            "indirect_only": [0, 1, 0],
            "no_path": [0, 0, 1],
        }

    def test_path_count_matches_mpc_evolution(self) -> None:
        """mpc_evolution should equal the valid path count per frame."""
        frame_5 = _make_frame(n_paths=5, frame_index=0)
        frame_10 = _make_frame(n_paths=10, frame_index=1)
        stats = self._collect([frame_5, frame_10])

        assert stats["mpc_evolution"] == [5, 10]

    def test_strongest_single_path_loss_is_minimum(self) -> None:
        """The strongest single-path loss is the minimum valid path loss."""
        path_loss = np.array([80.0, 60.0, 90.0, 55.0, 70.0], dtype=np.float64)
        frame = _make_frame(n_paths=5, path_loss_db=path_loss)
        stats = self._collect([frame])

        assert len(stats["strongest_single_path_loss_evolution"]) == 1
        assert stats["strongest_single_path_loss_evolution"][0] == pytest.approx(55.0, abs=0.1)

    def test_strongest_single_path_loss_is_nan_for_empty_frame(self) -> None:
        """The strongest single-path loss is unavailable for an empty frame."""
        frame = _make_frame(n_paths=0)
        stats = self._collect([frame])

        assert len(stats["strongest_single_path_loss_evolution"]) == 1
        assert np.isnan(stats["strongest_single_path_loss_evolution"][0])

    def test_frame_indices_tracked(self) -> None:
        """frame_indices should record the index of each processed frame."""
        stats = self._collect(
            [
                _make_frame(n_paths=3, frame_index=5),
                _make_frame(n_paths=3, frame_index=10),
                _make_frame(n_paths=3, frame_index=15),
            ]
        )

        assert stats["frame_indices"] == [5, 10, 15]


class TestTrajectoryMetricMapping:
    """Test the frame-index-to-metric mapping used by trajectory coloring."""

    def test_build_frame_index_to_metric_returns_dict(self) -> None:
        """_build_frame_index_to_metric should build a correct mapping."""
        from visualizer.src.panels.trajectory_preview_panel import (
            TrajectoryPreviewPanel,
        )

        # Create a minimal panel instance without a real parent
        class _FakeParent:
            trajectory_load_coordinator = TrajectoryLoadCoordinator()

        panel = TrajectoryPreviewPanel(_FakeParent())
        panel._per_frame_stats = {
            "frame_indices": [0, 1, 2, 3],
            "direct_path_pair_share_evolution": [1.0, 0.0, 0.5, 0.25],
            "delay_spread_evolution": [10.0, 20.0, 15.0, 25.0],
            "mpc_evolution": [100, 200, 150, 250],
            "strongest_single_path_loss_evolution": [55.0, 60.0, 58.0, 62.0],
        }

        direct_share_map = panel._build_frame_index_to_metric("direct_path_pair_share_evolution")
        assert direct_share_map is not None
        assert direct_share_map[0] == 1.0
        assert direct_share_map[2] == 0.5

        ds_map = panel._build_frame_index_to_metric("delay_spread_evolution")
        assert ds_map is not None
        assert ds_map[2] == pytest.approx(15.0)

        mpc_map = panel._build_frame_index_to_metric("mpc_evolution")
        assert mpc_map is not None
        assert mpc_map[3] == 250

        pl_map = panel._build_frame_index_to_metric("strongest_single_path_loss_evolution")
        assert pl_map is not None
        assert pl_map[0] == pytest.approx(55.0)

    def test_build_frame_index_to_metric_returns_none_without_stats(self) -> None:
        """Should return None when no per-frame stats are available."""
        from visualizer.src.panels.trajectory_preview_panel import (
            TrajectoryPreviewPanel,
        )

        class _FakeParent:
            trajectory_load_coordinator = TrajectoryLoadCoordinator()

        panel = TrajectoryPreviewPanel(_FakeParent())
        panel._per_frame_stats = None

        result = panel._build_frame_index_to_metric("direct_path_pair_share_evolution")
        assert result is None

    def test_build_frame_index_to_metric_returns_none_for_mismatched_lengths(
        self,
    ) -> None:
        """Should return None when frame_indices and values have different lengths."""
        from visualizer.src.panels.trajectory_preview_panel import (
            TrajectoryPreviewPanel,
        )

        class _FakeParent:
            trajectory_load_coordinator = TrajectoryLoadCoordinator()

        panel = TrajectoryPreviewPanel(_FakeParent())
        panel._per_frame_stats = {
            "frame_indices": [0, 1, 2],
            "direct_path_pair_share_evolution": [1.0, 0.0],
        }

        result = panel._build_frame_index_to_metric("direct_path_pair_share_evolution")
        assert result is None
