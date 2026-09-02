"""Pure statistics primitives for MPC frame analysis.

The functions re-exported here compute spreads, power-delay summaries,
histograms, and aggregate ``MPCStatistics`` records from arrays that callers
already loaded.
"""

from .distributions import (
    DEFAULT_BINS,
    compute_cdf,
    compute_histogram,
)
from .metrics import (
    compute_angular_spread,
    compute_binned_power_delay_profile,
    compute_delay_spread,
    compute_linear_angular_spread,
    compute_power_delay_profile,
    compute_signal_strength_distribution,
    compute_snr_distribution,
    pathloss_db_to_power_linear,
    power_linear_to_db,
)
from .mpc_statistics import (
    MPCStatistics,
    compute_comparison_stats,
    compute_mpc_statistics,
)

__all__ = [
    # Metrics
    "compute_delay_spread",
    "compute_angular_spread",
    "compute_binned_power_delay_profile",
    "compute_linear_angular_spread",
    "compute_power_delay_profile",
    "compute_signal_strength_distribution",
    "compute_snr_distribution",
    "pathloss_db_to_power_linear",
    "power_linear_to_db",
    # Distributions
    "compute_histogram",
    "compute_cdf",
    "DEFAULT_BINS",
    # MPC statistics
    "MPCStatistics",
    "compute_mpc_statistics",
    "compute_comparison_stats",
]
