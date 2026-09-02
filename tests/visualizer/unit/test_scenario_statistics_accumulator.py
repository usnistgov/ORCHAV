"""Semantic tests for selective scenario-statistics aggregation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

from shared.frames import FrameProjection, ProjectedMPCFrame
from shared.frames.contracts import (
    PATH_METRIC_ORDER,
    PATH_METRIC_VALIDITY_BITS,
    FrameReadRequest,
    PathMetric,
)
from shared.statistics.core.metrics import compute_delay_spread
from tests.visualizer.fixtures.semantic_mpc import (
    BASELINE_SEMANTIC_FRAME,
    CHANGED_SEMANTIC_FRAME,
    EMPTY_SEMANTIC_FRAME,
    SemanticMPCFrame,
)
from visualizer.src.metrics.scenario_statistics import (
    SCENARIO_STATISTICS_REQUEST,
    SCENARIO_STATISTICS_SCHEMA_VERSION,
    UNKNOWN_MPC_TYPE,
    ScenarioStatisticsAccumulator,
)

_SEMANTIC_METRIC_ATTRIBUTES = {
    PathMetric.DELAY_NS: "delay_ns",
    PathMetric.PATH_LOSS_DB: "path_loss_db",
    PathMetric.AOA_AZ_DEG: "aoa_az_deg",
    PathMetric.AOA_EL_DEG: "aoa_el_deg",
    PathMetric.AOD_AZ_DEG: "aod_az_deg",
    PathMetric.AOD_EL_DEG: "aod_el_deg",
}


def _semantic_projection(spec: SemanticMPCFrame, frame_index: int) -> FrameProjection:
    """Project a semantic fixture without using storage implementation code."""

    pair_counts = np.asarray([len(paths) for paths in spec.paths_by_pair], dtype=np.int64)
    pair_path_offsets = np.zeros((len(pair_counts) + 1,), dtype=np.int64)
    pair_path_offsets[1:] = np.cumsum(pair_counts)

    paths = spec.paths
    bounce_counts = np.asarray([len(path.bounces) for path in paths], dtype=np.int64)
    bounce_offsets = np.zeros((len(paths) + 1,), dtype=np.int64)
    bounce_offsets[1:] = np.cumsum(bounce_counts)
    interactions = np.asarray(
        [interaction for path in paths for interaction in path.interactions],
        dtype=np.uint8,
    )

    metric_values: dict[PathMetric, np.ndarray] = {}
    metric_valid_bits = np.zeros((len(paths),), dtype=np.uint8)
    for metric, attribute in _SEMANTIC_METRIC_ATTRIBUTES.items():
        values = np.asarray([getattr(path, attribute) for path in paths], dtype=np.float32)
        valid = np.isfinite(values)
        values[~valid] = np.nan
        metric_values[metric] = values
        metric_valid_bits[valid] |= np.uint8(PATH_METRIC_VALIDITY_BITS[metric])

    frame = ProjectedMPCFrame(
        frame_index=frame_index,
        tx_rx_pairs=np.asarray(spec.pair_order, dtype=np.int32),
        pair_path_offsets=pair_path_offsets,
        bounce_offsets=bounce_offsets,
        interactions=interactions,
        delays_ns=metric_values[PathMetric.DELAY_NS],
        path_loss_db=metric_values[PathMetric.PATH_LOSS_DB],
        aoa_az_deg=metric_values[PathMetric.AOA_AZ_DEG],
        aoa_el_deg=metric_values[PathMetric.AOA_EL_DEG],
        aod_az_deg=metric_values[PathMetric.AOD_AZ_DEG],
        aod_el_deg=metric_values[PathMetric.AOD_EL_DEG],
        metric_valid_bits=metric_valid_bits,
    )
    return FrameProjection.from_request(frame, SCENARIO_STATISTICS_REQUEST)


def _metric_projection(
    *,
    delays_ns: Iterable[float],
    path_loss_db: Iterable[float],
    frame_index: int = 0,
    interaction_orders: Iterable[int] | None = None,
) -> FrameProjection:
    """Build a LoS-only projection with deliberately asymmetric metric validity."""

    delays = np.asarray(tuple(delays_ns), dtype=np.float32)
    losses = np.asarray(tuple(path_loss_db), dtype=np.float32)
    if delays.shape != losses.shape:
        raise ValueError("Test delay and loss vectors must have equal shape")
    path_count = len(delays)
    orders = (
        np.zeros((path_count,), dtype=np.int64)
        if interaction_orders is None
        else np.asarray(tuple(interaction_orders), dtype=np.int64)
    )
    if orders.shape != (path_count,) or np.any(orders < 0):
        raise ValueError("Test interaction orders must be non-negative and path aligned")
    bounce_offsets = np.zeros((path_count + 1,), dtype=np.int64)
    bounce_offsets[1:] = np.cumsum(orders)
    interactions = np.ones((int(bounce_offsets[-1]),), dtype=np.uint8)

    metric_valid_bits = np.zeros((path_count,), dtype=np.uint8)
    delay_valid = np.isfinite(delays)
    loss_valid = np.isfinite(losses)
    metric_valid_bits[delay_valid] |= np.uint8(PATH_METRIC_VALIDITY_BITS[PathMetric.DELAY_NS])
    metric_valid_bits[loss_valid] |= np.uint8(PATH_METRIC_VALIDITY_BITS[PathMetric.PATH_LOSS_DB])

    angles: dict[PathMetric, np.ndarray] = {}
    for metric in (
        PathMetric.AOA_AZ_DEG,
        PathMetric.AOA_EL_DEG,
        PathMetric.AOD_AZ_DEG,
        PathMetric.AOD_EL_DEG,
    ):
        values = np.full((path_count,), np.nan, dtype=np.float32)
        if path_count:
            values[0] = 0.0
            metric_valid_bits[0] |= np.uint8(PATH_METRIC_VALIDITY_BITS[metric])
        angles[metric] = values

    frame = ProjectedMPCFrame(
        frame_index=frame_index,
        tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int32),
        pair_path_offsets=np.asarray([0, path_count], dtype=np.int64),
        bounce_offsets=bounce_offsets,
        interactions=interactions,
        delays_ns=delays,
        path_loss_db=losses,
        aoa_az_deg=angles[PathMetric.AOA_AZ_DEG],
        aoa_el_deg=angles[PathMetric.AOA_EL_DEG],
        aod_az_deg=angles[PathMetric.AOD_AZ_DEG],
        aod_el_deg=angles[PathMetric.AOD_EL_DEG],
        metric_valid_bits=metric_valid_bits,
    )
    return FrameProjection.from_request(frame, SCENARIO_STATISTICS_REQUEST)


def test_accumulator_matches_semantic_frame_sequence() -> None:
    """Aggregation preserves path, topology, metric, and interaction semantics."""

    specs = (
        BASELINE_SEMANTIC_FRAME,
        CHANGED_SEMANTIC_FRAME,
        EMPTY_SEMANTIC_FRAME,
    )
    projections = tuple(
        _semantic_projection(spec, frame_index) for frame_index, spec in enumerate(specs)
    )

    progress: list[tuple[int, int]] = []
    stats = ScenarioStatisticsAccumulator().collect_from_projections(
        iter(projections),
        total_frames=len(projections),
        on_progress=lambda current, total: progress.append((current, total)),
    )

    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert stats["total_mpcs"] == 7
    assert stats["total_frames"] == 3
    assert stats["reflection_order_dist"] == {0: 2, 1: 3, 2: 1, 3: 1}
    assert stats["mpc_evolution"] == [4, 3, 0]
    assert stats["frame_indices"] == [0, 1, 2]
    assert stats["pair_visibility_evolution"] == {
        "direct_path_present": [1, 1, 0],
        "indirect_only": [2, 2, 0],
        "no_path": [1, 1, 4],
    }
    assert stats["pair_visibility_counts"] == {
        "direct_path_present": 2,
        "indirect_only": 4,
        "no_path": 6,
    }
    np.testing.assert_allclose(
        stats["direct_path_pair_share_evolution"],
        np.asarray([0.25, 0.25, 0.0]),
    )
    assert stats["unique_tx_count"] == 2
    assert stats["unique_rx_count"] == 2
    assert stats["unique_tx_rx_pairs"] == 4
    assert set(stats["tx_rx_pairs"]) == set(BASELINE_SEMANTIC_FRAME.pair_order)

    # Validity bits retain measured zero and omit only unavailable values.
    np.testing.assert_allclose(
        stats["path_loss_values"],
        np.asarray([0.0, 41.0, 52.0, 63.0, 47.0, 33.0, 69.0]),
    )
    np.testing.assert_allclose(
        stats["delay_values"],
        np.asarray([0.0, 11.0, 23.0, 17.0, 3.0, 29.0]),
    )
    assert stats["path_loss_stats"]["min"] == 0.0
    np.testing.assert_allclose(
        stats["strongest_single_path_loss_evolution"],
        np.asarray([0.0, 33.0, np.nan]),
        equal_nan=True,
    )
    assert stats["pair_aggregate_path_gain_stats"]["count"] == 6
    assert stats["pair_rms_delay_spread_stats"]["count"] == 5
    assert stats["pair_visibility_summary"]["no_path"] == {
        "count": 6,
        "percent": 50.0,
    }

    # Virtual paths and unknown first-bounce codes have distinct buckets.
    assert stats["mpc_type_dist"] == {
        0: 2,
        1: 1,
        2: 1,
        4: 1,
        99: 1,
        UNKNOWN_MPC_TYPE: 1,
    }
    assert stats["mpc_type_evolution_per_frame"][99] == [1, 0, 0]
    assert stats["mpc_type_evolution_per_frame"][UNKNOWN_MPC_TYPE] == [0, 1, 0]


def test_validity_bits_retain_zero_and_jointly_align_delay_spread_inputs() -> None:
    """Zero is valid, while delay spread pairs only co-valid delay/loss rows."""

    projection = _metric_projection(
        delays_ns=(0.0, np.nan, 5.0),
        path_loss_db=(0.0, 10.0, np.nan),
    )

    stats = ScenarioStatisticsAccumulator().collect_from_projections([projection])

    assert stats["total_mpcs"] == 3
    np.testing.assert_array_equal(stats["delay_values"], np.asarray([0.0, 5.0]))
    np.testing.assert_array_equal(stats["path_loss_values"], np.asarray([0.0, 10.0]))
    np.testing.assert_array_equal(stats["aoa_az_values"], np.asarray([0.0]))
    assert stats["delay_stats"]["min"] == 0.0
    assert stats["path_loss_stats"]["min"] == 0.0
    assert stats["strongest_single_path_loss_evolution"] == [0.0]

    # Only path zero has both metrics.  Independently compacting the two arrays
    # would incorrectly pair delay=5 with loss=10 and produce a non-zero spread.
    assert stats["delay_spread_evolution"] == [0.0]
    assert stats["overall_delay_spread"] == 0.0
    assert stats["pair_aggregate_path_gain_db_values"][0] == pytest.approx(10.0 * np.log10(1.1))
    assert stats["pair_rms_delay_spread_ns_values"] == pytest.approx([0.0])


def test_no_covalid_delay_loss_is_nan_per_frame_and_absent_overall() -> None:
    projection = _metric_projection(
        delays_ns=(0.0, np.nan),
        path_loss_db=(np.nan, 10.0),
    )

    stats = ScenarioStatisticsAccumulator().collect_from_projections([projection])

    assert np.isnan(stats["delay_spread_evolution"][0])
    assert stats["overall_delay_spread"] is None


def test_interaction_orders_six_and_above_share_the_six_plus_bucket() -> None:
    projection = _metric_projection(
        delays_ns=(1.0, 2.0, 3.0, 4.0, 5.0),
        path_loss_db=(20.0, 20.0, 20.0, 20.0, 20.0),
        interaction_orders=(0, 5, 6, 7, 9),
    )

    stats = ScenarioStatisticsAccumulator().collect_from_projections([projection])

    assert SCENARIO_STATISTICS_SCHEMA_VERSION == 3
    assert stats["reflection_order_dist"] == {0: 1, 5: 1, 6: 3}
    assert stats["reflection_order_evolution_per_frame"][6] == [3]


def test_streamed_metrics_keep_compact_precision_until_float64_finalization() -> None:
    """Streaming avoids duplicate float64 chunks without changing public arrays."""

    projection = _metric_projection(
        delays_ns=(1.0, 2.0, 3.0),
        path_loss_db=(30.0, 40.0, 50.0),
    )
    accumulator = ScenarioStatisticsAccumulator()
    accumulator.add_projection(projection)

    for key in (
        "path_loss_values",
        "delay_values",
        "aoa_az_values",
        "aoa_el_values",
        "aod_az_values",
        "aod_el_values",
    ):
        chunks = accumulator._stats[key]
        assert len(chunks) == 1
        assert chunks[0].dtype == np.dtype(np.float32)

    # Common complete frames need no second retained delay/loss payload.
    assert accumulator._joint_delay_chunks[0] is accumulator._stats["delay_values"][0]
    assert accumulator._joint_path_loss_chunks[0] is accumulator._stats["path_loss_values"][0]

    stats = accumulator.finalize()
    for key in (
        "path_loss_values",
        "delay_values",
        "aoa_az_values",
        "aoa_el_values",
        "aod_az_values",
        "aod_el_values",
    ):
        assert stats[key].dtype == np.dtype(np.float64)


def test_asymmetric_metric_validity_preserves_aggregate_delay_spread() -> None:
    """Fallback joint vectors retain exact cross-frame delay/loss alignment."""

    projections = (
        _metric_projection(
            delays_ns=(0.0, np.nan, 5.0),
            path_loss_db=(0.0, 10.0, np.nan),
            frame_index=0,
        ),
        _metric_projection(
            delays_ns=(10.0, 20.0),
            path_loss_db=(20.0, 30.0),
            frame_index=1,
        ),
    )

    stats = ScenarioStatisticsAccumulator().collect_from_projections(projections)
    expected_delays = np.asarray([0.0, 10.0, 20.0], dtype=np.float64)
    expected_losses = np.asarray([0.0, 20.0, 30.0], dtype=np.float64)
    expected = compute_delay_spread(
        expected_delays,
        10.0 ** (-expected_losses / 10.0),
    )

    assert stats["overall_delay_spread"] == expected


def test_accumulator_rejects_projection_without_interactions() -> None:
    """The calculation fails explicitly instead of silently inventing path types."""

    complete = _metric_projection(delays_ns=(1.0,), path_loss_db=(20.0,))
    metrics_only_request = FrameReadRequest.for_metrics(PATH_METRIC_ORDER)
    incomplete_frame = ProjectedMPCFrame(
        frame_index=complete.frame.frame_index,
        tx_rx_pairs=complete.frame.tx_rx_pairs,
        pair_path_offsets=complete.frame.pair_path_offsets,
        bounce_offsets=complete.frame.bounce_offsets,
        delays_ns=complete.frame.delays_ns,
        path_loss_db=complete.frame.path_loss_db,
        aoa_az_deg=complete.frame.aoa_az_deg,
        aoa_el_deg=complete.frame.aoa_el_deg,
        aod_az_deg=complete.frame.aod_az_deg,
        aod_el_deg=complete.frame.aod_el_deg,
        metric_valid_bits=complete.frame.metric_valid_bits,
    )
    incomplete = FrameProjection.from_request(incomplete_frame, metrics_only_request)

    with pytest.raises(ValueError, match="path_interactions"):
        ScenarioStatisticsAccumulator().add_projection(incomplete)
