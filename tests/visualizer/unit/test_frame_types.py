"""Unit tests for frame_types helpers."""

from __future__ import annotations

import numpy as np

from shared.statistics import FrameStats, compute_orders_hist


class TestComputeOrdersHist:
    def test_from_dict(self):
        src = {0: 3, 1: 0, 2: 5, 4: 1}
        hist = compute_orders_hist(src)
        assert hist == {0: 3, 2: 5, 4: 1}

    def test_from_array(self):
        arr = np.array([2, 0, 4, 0, 1])
        hist = compute_orders_hist(arr)
        assert hist == {0: 2, 2: 4, 4: 1}


def test_frame_stats_dataclass_basics():
    stats = FrameStats(
        total_paths=10,
        orders_hist={0: 3, 1: 7},
        delay_range_ns=(1.0, 5.0),
        path_loss_range=(70.0, 120.0),
    )

    assert stats.total_paths == 10
    assert stats.orders_hist[1] == 7
    assert stats.delay_range_ns == (1.0, 5.0)
    assert stats.path_loss_range == (70.0, 120.0)
