"""Shared statistics API for frame and array analysis.

This package collects pure metric computations, distribution helpers, frame
summaries, material plots, and figure theme selection used across generator
summaries, analysis scripts, and visualization diagnostics. It operates on
frame data and arrays; it does not load scenarios or own frame I/O.
"""

from .core.distributions import (
    DEFAULT_BINS,
    compute_cdf,
    compute_histogram,
    compute_percentile,
    compute_statistics_summary,
)
from .core.metrics import (
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
from .core.mpc_statistics import (
    MPCStatistics,
    compute_comparison_stats,
    compute_mpc_statistics,
)
from .frame_stats import (
    FrameStats,
    compute_orders_hist,
)


def __getattr__(name):
    """Load visual helpers only when requested by a plotting consumer.

    Theme/material modules use frame metadata and its optional HDF5 stack.
    Pure array analysis and compact link exports must not require frame I/O.
    Existing public names remain available through their original import API.
    """
    if name == "theme_manager":
        from .themes import theme_manager

        globals()[name] = theme_manager
        return theme_manager
    if name in {
        "compute_material_counts_by_depth",
        "plot_material_by_bounce_depth",
        "plot_material_evolution_stacked",
    }:
        from . import viz

        value = getattr(viz, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "compute_percentile",
    "compute_statistics_summary",
    "DEFAULT_BINS",
    # MPC statistics
    "MPCStatistics",
    "compute_mpc_statistics",
    "compute_comparison_stats",
    # Themes
    "theme_manager",
    # Visualization
    "plot_material_evolution_stacked",
    "plot_material_by_bounce_depth",
    "compute_material_counts_by_depth",
    # Frame stats helpers
    "FrameStats",
    "compute_orders_hist",
]
