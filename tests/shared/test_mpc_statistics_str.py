"""
Tests for MPCStatistics.__str__() human-readable formatter.
"""

from shared.statistics.core.mpc_statistics import MPCStatistics


class TestMPCStatisticsStr:
    """Tests for MPCStatistics.__str__()."""

    def test_basic_output(self):
        """String output should include key metrics."""
        stats = MPCStatistics(
            P_incoh=0.001,
            P_incoh_db=-30.0,
            sigma_tau_ns=5.42,
            mean_delay_ns=12.3,
            max_delay_ns=25.0,
            min_delay_ns=3.0,
            N_paths=42,
            earliest_path_delay_ns=3.0,
            earliest_path_loss_db=55.0,
            rms_aoa_az_deg=15.2,
            rms_aoa_el_deg=8.1,
            mean_aoa_az_deg=45.0,
            mean_aoa_el_deg=10.0,
            rms_aod_az_deg=12.3,
            rms_aod_el_deg=6.5,
            mean_aod_az_deg=-30.0,
            mean_aod_el_deg=5.0,
        )
        s = str(stats)
        assert "MPC Statistics" in s
        assert "Paths:       42" in s
        assert "-30.0 dB" in s
        assert "spread=5.42 ns" in s
        assert "Earliest:" in s
        assert "delay=3.0 ns" in s
        assert "AoA spread:" in s
        assert "AoD spread:" in s

    def test_no_earliest_path(self):
        """String output omits the earliest-path line for an empty path set."""
        stats = MPCStatistics(
            P_incoh=0.001,
            P_incoh_db=-30.0,
            sigma_tau_ns=5.0,
            mean_delay_ns=12.0,
            max_delay_ns=25.0,
            min_delay_ns=3.0,
            N_paths=10,
            earliest_path_delay_ns=None,
            earliest_path_loss_db=None,
            rms_aoa_az_deg=0.0,
            rms_aoa_el_deg=0.0,
            mean_aoa_az_deg=0.0,
            mean_aoa_el_deg=0.0,
            rms_aod_az_deg=0.0,
            rms_aod_el_deg=0.0,
            mean_aod_az_deg=0.0,
            mean_aod_el_deg=0.0,
        )
        s = str(stats)
        assert "Earliest:" not in s

    def test_no_angular(self):
        """String output should omit angular lines when spreads are zero."""
        stats = MPCStatistics(
            P_incoh=0.001,
            P_incoh_db=-30.0,
            sigma_tau_ns=5.0,
            mean_delay_ns=12.0,
            max_delay_ns=25.0,
            min_delay_ns=3.0,
            N_paths=10,
            earliest_path_delay_ns=3.0,
            earliest_path_loss_db=55.0,
            rms_aoa_az_deg=0.0,
            rms_aoa_el_deg=0.0,
            mean_aoa_az_deg=0.0,
            mean_aoa_el_deg=0.0,
            rms_aod_az_deg=0.0,
            rms_aod_el_deg=0.0,
            mean_aod_az_deg=0.0,
            mean_aod_el_deg=0.0,
        )
        s = str(stats)
        assert "AoA spread:" not in s
        assert "AoD spread:" not in s
