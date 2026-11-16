"""Small display products selected by their authored sensing contracts."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from shared.sensing.products import select_product_by_contract


def product_kinds(sensing: Mapping, explicit=None) -> frozenset[str]:
    """An empty explicit inventory is authoritative, unlike an absent one."""
    if explicit is not None:
        return frozenset(explicit)
    return frozenset(
        descriptor.get("kind")
        for descriptor in sensing.get("products", {}).values()
        if isinstance(descriptor, Mapping) and descriptor.get("kind")
    )


def inspection_capabilities(kinds) -> dict[str, tuple[bool, str]]:
    """Describe missing signal evidence without depending on profile names."""
    kinds = frozenset(kinds)
    return {
        "cfar": (
            "range_doppler_detection_input_power" in kinds,
            "Detector-stage arrays were not recorded.",
        ),
        "clutter": ("range_doppler" in kinds, "Range–Doppler products were not recorded."),
        "inputs": (
            bool(
                kinds
                & {"delay_domain_slow_time", "delay_domain_power", "baseband_frequency_observation"}
            ),
            "CIR display power and complex observations were not recorded.",
        ),
        "frequency": (
            bool(kinds & {"baseband_frequency_observation", "baseband_frequency_power"}),
            "Frequency observations or display power were not recorded.",
        ),
        "angular": (
            "range_doppler_angle" in kinds,
            "This acquisition has no saved angular product.",
        ),
    }


def saved_micro_doppler(sensing: Mapping, variant: str = "primary") -> dict | None:
    """Read a fixed STFT without substituting an RD history or rerunning DSP."""
    if "micro_doppler" not in product_kinds(sensing):
        return None
    preferred = "primary" if variant in {"auto", "clean"} else variant
    variants = {
        d.get("variant")
        for d in sensing.get("products", {}).values()
        if d.get("kind") == "micro_doppler"
    }
    if preferred not in variants:
        # A recording can author only one mean policy; the descriptor remains
        # the displayed policy, irrespective of the ordinary RD variant choice.
        preferred = "primary" if "primary" in variants else "raw"
    _, values, axes, descriptor = select_product_by_contract(
        sensing,
        kind="micro_doppler",
        variant=preferred,
        axis_names=("pair", "slow_time", "doppler_frequency"),
        representations=("power_linear",),
    )
    if values.shape[0] != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Saved STFT requires one pair and finite nonnegative power")
    times, doppler = (np.asarray(axis, dtype=float) for axis in axes[1:])
    if any(not np.isfinite(a).all() or np.any(np.diff(a) <= 0) for a in (times, doppler)):
        raise ValueError("Saved STFT coordinates must be finite and increase")
    processing = descriptor.get("processing", {})
    recipe = processing.get("recipe", processing)
    hop = processing.get("hop_duration_s")
    if hop is None:
        period = processing.get("sample_interval_s")
        if period is None:
            period = (
                sensing.get("config_resolved", {}).get("acquisition", {}).get("sample_period_s")
            )
        if period is not None and recipe.get("hop_samples") is not None:
            hop = float(period) * int(recipe["hop_samples"])
    if hop is None or not np.isfinite(hop) or hop <= 0:
        raise ValueError("Saved STFT must declare a positive physical hop duration")
    return {
        "power": values[0],
        "times_s": times,
        "doppler_hz": doppler,
        "hop_s": float(hop),
        "recipe": dict(recipe),
        "variant": preferred,
    }


def saved_micro_doppler_unavailable_reason(sensing: Mapping) -> str:
    """Explain an authored gate or acquisition limit when it saved no columns."""
    reason = (sensing.get("display_status") or {}).get("micro_doppler")
    return (
        f"No saved STFT columns: {reason}."
        if reason
        else "A fixed STFT display product was not recorded."
    )


def micro_doppler_history(records, sensing: Mapping, variant: str) -> dict | None:
    """Join authored columns, preserving gaps and rejecting conflicting grids."""
    blocks = [
        {
            "power": r.micro_power,
            "times_s": r.micro_times_s,
            "doppler_hz": r.micro_doppler_hz,
            "hop_s": r.micro_hop_s,
            "recipe": r.micro_recipe,
        }
        for r in records
        if r.micro_power is not None
    ]
    if not blocks:
        block = saved_micro_doppler(sensing, variant)
        if block is None:
            return None
        blocks = [block]
    first = blocks[0]
    columns = {}
    for block in blocks:
        if (
            not np.array_equal(block["doppler_hz"], first["doppler_hz"])
            or block["recipe"] != first["recipe"]
        ):
            raise ValueError("Saved STFT grids or processing recipes differ within this recording")
        for time_s, power in zip(block["times_s"], block["power"], strict=True):
            previous = columns.get(float(time_s))
            if previous is not None and not np.array_equal(previous[0], power):
                raise ValueError("Conflicting saved STFT columns at the same physical epoch")
            columns[float(time_s)] = (power, block["hop_s"])
    times = sorted(columns)
    return {
        "times_s": np.asarray(times),
        "power": np.stack([columns[t][0] for t in times]),
        "widths_s": np.asarray([columns[t][1] for t in times]),
        "doppler_hz": first["doppler_hz"],
        "recipe": first["recipe"],
    }
