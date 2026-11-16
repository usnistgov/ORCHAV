"""Range-profile processing shared by sequential and analytic sensing strategies.

``compute_range_profile_block`` attaches power, optional clutter estimates,
and one-dimensional CFAR decisions to the caller's sensing payload.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shared.sensing.detection.cfar import cfar_1d
from shared.sensing.signal_processing import (
    compute_range_profile,
    sensing_window,
    window_energy_correction,
)


def compute_range_profile_block(
    processor: Any,
    a_link: np.ndarray,
    tau_link: np.ndarray,
    sensing_data: dict[str, Any],
    clutter_metadata: dict[str, Any],
    cfar_metadata: dict[str, Any],
    profiler: Any,
    *,
    acquired_samples: np.ndarray | None = None,
    compute_detections: bool = True,
) -> bool:
    """Compute range profile, apply clutter removal and 1-D CFAR.

    Populates *sensing_data*, *clutter_metadata*, and *cfar_metadata* in-place.
    When supplied, *acquired_samples* includes the already drawn sample-domain
    noise. Its tapered power feeds both history subtraction and CFAR; the path
    profile still supplies the common range-processing metadata.

    Returns:
        True if a range profile was computed, False otherwise.
    """
    if not (processor.config.output_range_profile or processor.config.output_detections):
        return False

    with profiler.stage("range_profile"):
        rp_result = compute_range_profile(
            a_link,
            tau_link,
            sampling_frequency=processor.config.bandwidth,
            fft_size=processor.config.fft_size_range,
            window=processor.config.window_range,
            range_mode=processor.range_mode,
        )
    rp_power = rp_result.power_linear
    if acquired_samples is not None:
        taper = sensing_window(processor.config.window_range, processor.config.fft_size_range)
        rp_power = np.abs(acquired_samples * taper * window_energy_correction(taper)) ** 2
    if rp_result.metadata:
        sensing_data.update(rp_result.metadata)
        for config_key in ("config", "config_resolved"):
            config = sensing_data.get(config_key)
            if isinstance(config, dict):
                config.update(rp_result.metadata)
    if processor._should_emit_raw_payload("range_profile_raw"):
        sensing_data["range_profile_raw"] = rp_power.copy()

    baseline_info: np.ndarray | None = None
    if processor._range_clutter is not None:
        removal = processor._range_clutter.process(rp_power)
        rp_power = removal.cleaned
        baseline_info = removal.baseline
        clutter_metadata["range_profile"] = {
            "enabled": True,
            "window_frames": processor.clutter_window,
            "baseline_frames": removal.frames_used,
            "method": "causal_previous_sample_mean_subtraction",
            **removal.processing,
        }

    if compute_detections and processor.config.cfar_ref_cells > 0:
        range_cfar_result = cfar_1d(
            rp_power,
            guard_cells=processor.config.cfar_guard_cells,
            reference_cells=processor.config.cfar_ref_cells,
            threshold_scale=processor.config.cfar_threshold_scale,
            method=processor.config.cfar_type.lower(),  # type: ignore[arg-type]
            os_rank=processor.config.cfar_os_rank,
            edge_policy=processor.config.cfar_edge_policy,
        )
        sensing_data["range_profile_cfar_mask"] = range_cfar_result.detections.astype(np.uint8)
        cfar_metadata["range_profile"] = {
            "method": processor.config.cfar_type.lower(),
            "guard_cells": processor.config.cfar_guard_cells,
            "reference_cells": processor.config.cfar_ref_cells,
            "threshold_scale": processor.config.cfar_threshold_scale,
            "threshold_semantics": "raw_linear_power_multiplier",
            "os_rank": processor.config.cfar_os_rank,
            "os_rank_rounding": "round_fraction_times_k_minus_one",
            "edge_policy": processor.config.cfar_edge_policy,
            "cut_comparison": "strict_greater_than_threshold",
            "eligible_cell_count": int(np.count_nonzero(range_cfar_result.eligible)),
            "training_cell_count_min": (
                int(np.min(range_cfar_result.training_cell_count[range_cfar_result.eligible]))
                if np.any(range_cfar_result.eligible)
                else None
            ),
            "training_cell_count_max": (
                int(np.max(range_cfar_result.training_cell_count[range_cfar_result.eligible]))
                if np.any(range_cfar_result.eligible)
                else None
            ),
        }

    if processor.config.output_range_profile:
        sensing_data["range_profile"] = rp_power
        sensing_data["range_profile_kind"] = "power_linear"
        if baseline_info is not None and processor._should_emit_raw_payload(
            "range_profile_baseline"
        ):
            sensing_data["range_profile_baseline"] = baseline_info
    elif processor.config.output_detections:
        sensing_data["range_profile_kind"] = "power_linear"

    return True
