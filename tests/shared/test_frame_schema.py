"""Tests for canonical frame summary helpers."""

import numpy as np

from shared.frames import PathMetric, StandardMPCFrame, standard_mpc_frame_from_pair_data
from shared.frames.schema import count_frame_mpcs, summarize_frame


def _sample_frame(*, pair_count: int = 2) -> StandardMPCFrame:
    pairs = [[0, index] for index in range(pair_count)]
    vertices = [np.zeros((5 + 5 * index, 1, 3), dtype=np.float32) for index in range(pair_count)]
    interactions = [np.ones((5 + 5 * index, 1), dtype=np.uint8) for index in range(pair_count)]
    return standard_mpc_frame_from_pair_data(
        frame_index=0,
        tx_rx_pairs=pairs,
        tx_positions=[[1.0, 2.0, 3.0]],
        rx_positions=[[4.0 + index, 5.0, 6.0] for index in range(pair_count)],
        vertices_by_pair=vertices,
        interactions_by_pair=interactions,
        metrics_by_pair={
            PathMetric.DELAY_NS: [
                np.arange(5 + 5 * index, dtype=np.float32) for index in range(pair_count)
            ],
            PathMetric.PATH_LOSS_DB: [
                np.arange(5 + 5 * index, dtype=np.float32) + 60.0 for index in range(pair_count)
            ],
        },
    )


def test_count_frame_mpcs_uses_compact_offsets() -> None:
    assert count_frame_mpcs(_sample_frame()) == 15


def test_summary_contains_counts_paths_and_valid_metrics() -> None:
    summary = summarize_frame(_sample_frame())

    assert "TX: 1" in summary
    assert "RX: 2" in summary
    assert "Pairs: 2" in summary
    assert "Total MPCs: 15" in summary
    assert "Paths per pair: [5, 10]" in summary
    assert "delays_ns" in summary
    assert "path_loss_db" in summary


def test_summary_reports_absent_metrics() -> None:
    frame = standard_mpc_frame_from_pair_data(
        frame_index=0,
        tx_rx_pairs=[],
        tx_positions=[],
        rx_positions=[],
        vertices_by_pair=[],
        interactions_by_pair=[],
    )

    summary = summarize_frame(frame)

    assert "TX: 0" in summary
    assert "Total MPCs: 0" in summary
    assert "Metrics: none" in summary


def test_summary_notes_extensions() -> None:
    frame = _sample_frame()
    values = {
        field: getattr(frame, field) for field in frame.__dataclass_fields__ if field != "version"
    }
    values.update(
        beamforming={"weights": [1, 0]},
        sensing={"range_profile": np.asarray([0.1, 0.2])},
    )
    extended = StandardMPCFrame(**values)
    summary = summarize_frame(extended)

    assert "Beamforming: yes" in summary
    assert "Frame extension: sensing" in summary


def test_summary_compacts_large_pair_lists() -> None:
    assert "11 pairs, see data" in summarize_frame(_sample_frame(pair_count=11))
