from types import SimpleNamespace

import numpy as np

from visualizer.src.metrics.mpc_stats import MPCStatsComputer


def test_mpc_stats_uses_path_level_arrays_and_mask():
    canon = SimpleNamespace(
        path_orders=np.array([0, 2], dtype=np.int64),
        path_delays=np.array([10.0, 20.0], dtype=float),
        path_losses=np.array([80.0, 90.0], dtype=float),
        path_tx=np.array([0, 1], dtype=np.int64),
        path_rx=np.array([0, 1], dtype=np.int64),
        path_start_indices=np.array([0, 2], dtype=np.int64),
        path_id=np.array([0, 0, 1, 1], dtype=np.int64),
        aod_az=np.array([5.0, 5.0, 55.0, 55.0], dtype=float),
    )

    stats = MPCStatsComputer().compute_frame_stats(
        canon,
        include_advanced=True,
        path_mask=np.array([False, True], dtype=bool),
    )

    assert stats.total_paths == 1
    assert stats.orders_hist == {2: 1}
    assert stats.delay_range_ns == (20.0, 20.0)
    assert stats.path_loss_range == (90.0, 90.0)
    assert not hasattr(stats, "power_delay_profile")
    assert stats.binned_power_delay_profile is not None
    np.testing.assert_allclose(
        stats.binned_power_delay_profile,
        (np.array([20.0]), np.array([-90.0])),
    )


def test_mpc_stats_aligns_real_delay_and_loss_without_estimated_fallbacks():
    canon = SimpleNamespace(
        path_orders=np.array([0, 1, 2, 3], dtype=np.int64),
        path_delays=np.array([10.0, 20.0, np.nan, 40.0]),
        path_losses=np.array([80.0, np.nan, 100.0, 110.0]),
        path_delay_is_estimated=np.array([False, False, False, True]),
        path_loss_is_estimated=np.array([False, False, False, False]),
        path_aod_az=np.array([0.0, 20.0, 40.0, 60.0]),
        path_tx=np.zeros(4, dtype=np.int64),
        path_rx=np.zeros(4, dtype=np.int64),
    )

    stats = MPCStatsComputer().compute_frame_stats(canon, include_advanced=True)

    assert stats.total_paths == 4
    assert stats.delay_range_ns == (10.0, 20.0)
    assert stats.path_loss_range == (80.0, 110.0)
    assert stats.binned_power_delay_profile is not None
    np.testing.assert_array_equal(stats.binned_power_delay_profile[0], np.array([10.0]))
    assert stats.delay_spread_ns == 0.0
    assert stats.angular_spread_deg > 0.0


def test_mpc_stats_rejects_misaligned_selection_mask():
    canon = SimpleNamespace(
        path_orders=np.array([0, 1], dtype=np.int64),
        path_delays=np.array([10.0, 20.0]),
        path_losses=np.array([80.0, 90.0]),
    )

    stats = MPCStatsComputer().compute_frame_stats(
        canon,
        path_mask=np.array([True], dtype=bool),
    )

    assert stats.total_paths == 0
    assert stats.orders_hist == {}
