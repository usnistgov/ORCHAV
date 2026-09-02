"""Renderer-neutral policy and semantic content for the viewport HUD.

The visualizer panels own user intent and renderer backends own presentation.
This module keeps the small amount of policy between them independent from Qt
and pygfx so filter summaries and scalar legends can be unit-tested directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from ..state import (
    DEFAULT_MPC_ALLOWED_ORDERS,
    DEFAULT_MPC_ALLOWED_TYPES,
    ViewportHudMode,
    normalize_viewport_hud_mode,
)
from .mpc_interaction_style_service import mpc_interaction_label

_TRAJECTORY_LABELS: Mapping[str, tuple[str, str]] = {
    "speed": ("Trajectory Speed", "m/frame"),
    "altitude": ("Trajectory Altitude", "m"),
    "time": ("Trajectory Time", "frame"),
    "angular_speed": ("Trajectory Angular Speed", "rad/frame"),
}


@dataclass(frozen=True, slots=True)
class ViewportHudPolicy:
    """Effective HUD policy derived from application state."""

    enabled: bool
    mode: ViewportHudMode
    show_status: bool
    show_legends: bool
    show_filters: bool
    show_annotations: bool

    @property
    def detailed(self) -> bool:
        """Return whether detailed rather than compact content is requested."""
        return self.mode == "detailed"


def viewport_hud_policy(state: Any) -> ViewportHudPolicy:
    """Build the effective viewport-HUD policy from an AppState-like object."""
    raw_mode = getattr(state, "viewport_hud_mode", "compact")
    mode = normalize_viewport_hud_mode(raw_mode)
    enabled_value = getattr(state, "viewport_hud_enabled", None)
    enabled = (
        str(raw_mode or "").strip().lower() != "off"
        if enabled_value is None
        else bool(enabled_value)
    )
    return ViewportHudPolicy(
        enabled=enabled,
        mode=mode,
        show_status=enabled and bool(getattr(state, "viewport_hud_show_status", True)),
        show_legends=enabled and bool(getattr(state, "viewport_hud_show_legends", True)),
        show_filters=enabled and bool(getattr(state, "viewport_hud_show_filters", True)),
        show_annotations=enabled and bool(getattr(state, "viewport_hud_show_annotations", True)),
    )


@dataclass(frozen=True, slots=True)
class PathFilterSummary:
    """Compact and detailed representations of active MPC path filters."""

    details: tuple[str, ...]

    @property
    def active_count(self) -> int:
        """Return the number of independently active filter groups."""
        return len(self.details)

    @property
    def active(self) -> bool:
        """Return whether any path filter is active."""
        return bool(self.details)

    @property
    def compact_text(self) -> str:
        """Return a bounded summary appropriate for the compact HUD."""
        suffix = "filter" if self.active_count == 1 else "filters"
        return f"Paths filtered · {self.active_count} {suffix}"


def _compact_integer_ranges(values: Iterable[int]) -> str:
    """Return sorted integers as comma-separated singleton/range tokens."""
    sorted_values = sorted({int(value) for value in values})
    if not sorted_values:
        return "none"
    tokens: list[str] = []
    start = previous = sorted_values[0]
    for value in sorted_values[1:]:
        if value == previous + 1:
            previous = value
            continue
        tokens.append(str(start) if start == previous else f"{start}–{previous}")
        start = previous = value
    tokens.append(str(start) if start == previous else f"{start}–{previous}")
    return ", ".join(tokens)


def _bounded_range_detail(
    label: str,
    lower: Optional[float],
    upper: Optional[float],
    unit: str,
) -> Optional[str]:
    """Format one optional lower/upper numeric filter pair."""
    if lower is None and upper is None:
        return None
    if lower is None:
        return f"{label} ≤ {float(upper):g} {unit}".rstrip()
    if upper is None:
        return f"{label} ≥ {float(lower):g} {unit}".rstrip()
    return f"{label}: {float(lower):g}–{float(upper):g} {unit}".rstrip()


def build_path_filter_summary(
    state: Any,
    *,
    allowed_materials: Optional[Sequence[str]] = None,
) -> PathFilterSummary:
    """Describe active MPC filters without depending on panel widget state."""
    details: list[str] = []

    allowed_orders = frozenset(
        int(value) for value in getattr(state, "mpc_allowed_orders", DEFAULT_MPC_ALLOWED_ORDERS)
    )
    if allowed_orders != DEFAULT_MPC_ALLOWED_ORDERS:
        details.append(f"Orders: {_compact_integer_ranges(allowed_orders)}")

    allowed_types = frozenset(
        int(value) for value in getattr(state, "mpc_allowed_types", DEFAULT_MPC_ALLOWED_TYPES)
    )
    if allowed_types != DEFAULT_MPC_ALLOWED_TYPES:
        labels = [
            (
                mpc_interaction_label(value, compact=True)
                if value in DEFAULT_MPC_ALLOWED_TYPES
                else f"Type {value}"
            )
            for value in sorted(allowed_types)
        ]
        details.append(f"Types: {', '.join(labels) if labels else 'none'}")

    if allowed_materials is not None:
        material_names = tuple(str(value) for value in allowed_materials)
        details.append(
            "Materials: none"
            if not material_names
            else f"Materials: {len(material_names)} selected"
        )

    numeric_filters = (
        (
            "Delay",
            getattr(state, "delay_filter_min_ns", None),
            getattr(state, "delay_filter_max_ns", None),
            "ns",
        ),
        (
            "Path loss",
            getattr(state, "power_filter_min_db", None),
            getattr(state, "power_filter_max_db", None),
            "dB",
        ),
        (
            "AoA az",
            getattr(state, "aoa_az_filter_min_deg", None),
            getattr(state, "aoa_az_filter_max_deg", None),
            "°",
        ),
        (
            "AoA el",
            getattr(state, "aoa_el_filter_min_deg", None),
            getattr(state, "aoa_el_filter_max_deg", None),
            "°",
        ),
        (
            "AoD az",
            getattr(state, "aod_az_filter_min_deg", None),
            getattr(state, "aod_az_filter_max_deg", None),
            "°",
        ),
        (
            "AoD el",
            getattr(state, "aod_el_filter_min_deg", None),
            getattr(state, "aod_el_filter_max_deg", None),
            "°",
        ),
    )
    for label, lower, upper, unit in numeric_filters:
        detail = _bounded_range_detail(label, lower, upper, unit)
        if detail is not None:
            details.append(detail)

    if bool(getattr(state, "topk_render_enabled", False)):
        maximum = max(1, int(getattr(state, "topk_render_max_paths", 1)))
        details.append(f"Top paths: {maximum:,}")

    return PathFilterSummary(details=tuple(details))


@dataclass(frozen=True, slots=True)
class TrajectoryHudLegend:
    """Semantic scalar legend for a visible non-categorical trajectory mode."""

    title: str
    unit: str
    value_range: tuple[float, float]


def build_trajectory_hud_legend(
    color_mode: str,
    scalar_range: Optional[tuple[float, float]],
) -> Optional[TrajectoryHudLegend]:
    """Return a trajectory scalar legend, or ``None`` for categorical mode."""
    semantic = _TRAJECTORY_LABELS.get(str(color_mode).strip().lower())
    if semantic is None or scalar_range is None:
        return None
    try:
        lower, upper = (float(scalar_range[0]), float(scalar_range[1]))
    except (IndexError, TypeError, ValueError):
        return None
    if not math.isfinite(lower) or not math.isfinite(upper):
        return None
    if lower > upper:
        lower, upper = upper, lower
    title, unit = semantic
    return TrajectoryHudLegend(title=title, unit=unit, value_range=(lower, upper))
