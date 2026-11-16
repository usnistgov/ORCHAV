"""Frame and pulse timing metadata for emitted sensing products.

Scene-step timing determines revisits; the pulse repetition frequency (PRF)
determines sample spacing within a coherent processing interval (CPI).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict


def build_sensing_timing_metadata(
    *,
    step_idx: int,
    first_step: int,
    cir_steps: int,
    frame_dt_s: float,
    prf_hz: float,
) -> Dict[str, Any]:
    """Describe a frame's position within its CPI, with all durations in seconds.

    Nonpositive frame intervals use a one-second fallback. A nonpositive PRF
    uses the frame interval as the pulse spacing.
    """
    coherent_samples_per_cpi = max(1, int(cir_steps))
    safe_frame_dt = float(frame_dt_s) if frame_dt_s > 0 else 1.0
    safe_prf = float(prf_hz) if prf_hz > 0 else 0.0
    pri_s = (1.0 / safe_prf) if safe_prf > 0 else safe_frame_dt

    relative_step = max(0, int(step_idx) - int(first_step))
    cpi_index = relative_step // coherent_samples_per_cpi
    pri_index_in_cpi = relative_step % coherent_samples_per_cpi

    return {
        "frame_dt_s": safe_frame_dt,
        "pri_s": pri_s,
        "coherent_samples_per_cpi": coherent_samples_per_cpi,
        "cpi_duration_s": pri_s * coherent_samples_per_cpi,
        "revisit_interval_s": safe_frame_dt * coherent_samples_per_cpi,
        "cpi_index": int(cpi_index),
        "pri_index_in_cpi": int(pri_index_in_cpi),
        "timing_role": (
            "cpi_end"
            if coherent_samples_per_cpi > 1 and pri_index_in_cpi == coherent_samples_per_cpi - 1
            else "pri_sample"
        ),
    }


def merge_sensing_timing_metadata(
    base_timing: Mapping[str, Any],
    extra_timing: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Copy timing metadata, adding strategy fields without replacing base values."""
    merged = dict(base_timing)
    if not isinstance(extra_timing, Mapping):
        return merged

    for key, value in extra_timing.items():
        if key not in merged:
            merged[key] = value

    return merged
