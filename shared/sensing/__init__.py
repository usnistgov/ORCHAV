"""Reusable sensing contracts, numerical processing, and saved-result workflows.

Start with ``products``, ``loader``, and ``writer`` for persisted results,
``signal_processing`` and ``detection`` for numerical processing, and
``postprocessor`` for the offline workflow. ``fusion`` and ``volume`` own
their saved-result models; ``plotting`` provides shared scientific figures.

The established numerical API remains available directly from this package.
Exports are loaded on demand so importing a lightweight contract does not load
signal processing, plotting, or application runtimes.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "C0",
    "DELAY_PROJECTION_POINT_BIN_FLOOR",
    "DIRECT_DOPPLER_CLUTTER_FILTER_POLICY_DISABLED",
    "DIRECT_DOPPLER_CLUTTER_FILTER_POLICY_EXCLUDE_BELOW_CUTOFF",
    "DOPPLER_FFT_NORMALIZATION_UNITARY",
    "DetectionVelocityDisplay",
    "DopplerProfileResult",
    "GTEntry",
    "MicroDopplerResult",
    "RANGE_PROCESSING_DELAY_DOMAIN_CIR_BINS",
    "RangeDopplerResult",
    "RangeProfileResult",
    "RelativeComplexAwgnResult",
    "SLOW_TIME_MEAN_REMOVAL_POLICY_NONZERO_VARIATION",
    "SLOW_TIME_MEAN_REMOVAL_POLICY_WINDOWED",
    "WINDOW_NORMALIZATION_ENERGY",
    "apply_aoa_filter",
    "apply_detection_velocity_metadata",
    "apply_pre_cfar_doppler_guards",
    "auto_select_range_bin",
    "bistatic_bisector_velocity_from_doppler_m_s",
    "bistatic_path_closing_rate_m_s",
    "cir_to_samples",
    "cir_to_samples_batch",
    "cir_to_samples_per_antenna",
    "coherent_clutter_metadata",
    "compute_analytical_micro_doppler",
    "compute_collapsed_doppler_profile",
    "compute_gt_entry",
    "compute_micro_doppler",
    "compute_range_collapsed_micro_doppler",
    "compute_range_doppler",
    "compute_range_profile",
    "compute_range_tracked_micro_doppler",
    "compute_range_tracked_range_doppler",
    "detection_velocity_display",
    "doppler_axis_to_velocity_m_s",
    "doppler_to_monostatic_equivalent_velocity_m_s",
    "doppler_to_path_closing_rate_m_s",
    "doppler_to_velocity_m_s",
    "extract_cir_link",
    "extract_target_positions",
    "format_detection_velocity_m_s",
    "has_fresh_sensing_data",
    "inject_target_relative_complex_awgn",
    "normalize_range_mode",
    "project_cir_to_delay_bins",
    "radial_velocity_axis",
    "remove_window_weighted_mean",
    "sensing_window",
    "window_energy_correction",
]

# Existing callers also use these numerical aliases.
_COMPATIBILITY_EXPORTS = frozenset({"_range_axis", "_window", "_window_energy_correction"})


def __getattr__(name: str) -> Any:
    if name in __all__ or name in _COMPATIBILITY_EXPORTS:
        value = getattr(import_module(".signal_processing", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
