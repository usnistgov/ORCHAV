"""Canonical MPC interaction-type colors and legend semantics.

The MPC pipeline, control-panel legend, and viewport HUD all consume this
renderer-neutral mapping. Unknown values remain visibly distinct from known
interaction types, and virtual reconstruction points keep their authored
orange instead of being clipped into the diffraction palette slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

MPC_VIRTUAL_COLOR = (1.0, 0.6, 0.2)
MPC_UNKNOWN_COLOR = (0.5, 0.5, 0.5)


@dataclass(frozen=True, slots=True)
class MpcInteractionStyle:
    """Semantic and palette metadata for one MPC interaction type."""

    interaction_type: Optional[int]
    label: str
    compact_label: str
    palette_index: Optional[int]
    fixed_color: Optional[tuple[float, float, float]]
    tooltip: str


@dataclass(frozen=True, slots=True)
class MpcInteractionLegendEntry:
    """One fully resolved interaction-type legend row."""

    interaction_type: Optional[int]
    label: str
    compact_label: str
    color: tuple[float, float, float]
    tooltip: str


MPC_INTERACTION_STYLES: tuple[MpcInteractionStyle, ...] = (
    MpcInteractionStyle(
        0,
        "LoS",
        "LoS",
        0,
        None,
        "Direct component without touching any surface.",
    ),
    MpcInteractionStyle(
        1,
        "Specular",
        "Spec",
        1,
        None,
        "Mirror-like reflection on smooth surfaces.",
    ),
    MpcInteractionStyle(
        2,
        "Diffuse",
        "Diffuse",
        2,
        None,
        "Energy scattered by rough surfaces.",
    ),
    MpcInteractionStyle(
        4,
        "Refraction",
        "Refract",
        4,
        None,
        "Signal bending through dielectric materials.",
    ),
    MpcInteractionStyle(
        8,
        "Diffraction",
        "Diffract",
        5,
        None,
        "Edge or wedge diffraction around obstacles.",
    ),
    MpcInteractionStyle(
        99,
        "Virtual",
        "Virtual",
        None,
        MPC_VIRTUAL_COLOR,
        "Path reconstructed using virtual bounce points.",
    ),
)
UNKNOWN_MPC_INTERACTION_STYLE = MpcInteractionStyle(
    None,
    "Unknown",
    "Unknown",
    None,
    MPC_UNKNOWN_COLOR,
    "Unrecognized or unsupported interaction type.",
)
_MPC_INTERACTION_STYLE_BY_TYPE = {style.interaction_type: style for style in MPC_INTERACTION_STYLES}
_MPC_INTERACTION_SORT_INDEX = {
    style.interaction_type: index for index, style in enumerate(MPC_INTERACTION_STYLES)
}


def mpc_interaction_style(interaction_type: int) -> MpcInteractionStyle:
    """Return the canonical style for a raw interaction type."""
    return _MPC_INTERACTION_STYLE_BY_TYPE.get(
        int(interaction_type),
        UNKNOWN_MPC_INTERACTION_STYLE,
    )


def mpc_interaction_label(
    interaction_type: int,
    *,
    compact: bool = False,
    explicit_unknown: bool = False,
) -> str:
    """Return a canonical display label for a raw interaction type."""
    value = int(interaction_type)
    style = mpc_interaction_style(value)
    if explicit_unknown and style is UNKNOWN_MPC_INTERACTION_STYLE:
        return f"Unknown (Type {value})"
    return style.compact_label if compact else style.label


def mpc_interaction_sort_key(interaction_type: int) -> tuple[int, int]:
    """Return canonical known-type order followed by numeric unknown values."""
    value = int(interaction_type)
    return (
        _MPC_INTERACTION_SORT_INDEX.get(value, len(MPC_INTERACTION_STYLES)),
        value,
    )


def build_mpc_type_palette(
    interaction_colors: Mapping[int, Sequence[float]],
    *,
    size: int = 9,
) -> np.ndarray:
    """Build the indexed palette consumed by MPC colorization."""
    palette = np.empty((max(int(size), 6), 3), dtype=np.float32)
    palette[:] = MPC_UNKNOWN_COLOR
    for style in MPC_INTERACTION_STYLES:
        index = style.palette_index
        interaction_type = style.interaction_type
        if index is None or interaction_type is None:
            continue
        color = interaction_colors.get(interaction_type)
        if color is None:
            continue
        values = np.asarray(color, dtype=np.float32).reshape(-1)
        if values.size >= 3:
            palette[index] = np.clip(values[:3], 0.0, 1.0)
    return palette


def _resolved_style_color(
    style: MpcInteractionStyle,
    type_palette: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    """Resolve one semantic style against the active categorical palette."""
    if style.fixed_color is not None:
        return style.fixed_color
    palette = np.asarray(type_palette, dtype=np.float32)
    index = style.palette_index
    if (
        palette.ndim != 2
        or palette.shape[1] < 3
        or index is None
        or index < 0
        or index >= len(palette)
    ):
        return MPC_UNKNOWN_COLOR
    color = np.clip(palette[index, :3], 0.0, 1.0)
    return float(color[0]), float(color[1]), float(color[2])


def colorize_mpc_interaction_types(
    interaction_types: np.ndarray,
    type_palette: Sequence[Sequence[float]],
) -> np.ndarray:
    """Map raw MPC interaction values to RGB without aliasing unknown values."""
    raw = np.asarray(interaction_types)
    flat = raw.reshape(-1)
    colors = np.empty((flat.size, 3), dtype=np.float32)
    colors[:] = MPC_UNKNOWN_COLOR
    for style in MPC_INTERACTION_STYLES:
        colors[flat == style.interaction_type] = _resolved_style_color(style, type_palette)
    return colors.reshape((*raw.shape, 3))


def mpc_interaction_legend_entries(
    type_palette: Sequence[Sequence[float]],
    *,
    present_types: Optional[Iterable[int]] = None,
) -> tuple[MpcInteractionLegendEntry, ...]:
    """Return canonical interaction rows, optionally limited to present codes.

    Known rows require their exact interaction code. The single Unknown row is
    included when any supplied code is unsupported, regardless of how many
    distinct unsupported codes are present. Omitting ``present_types`` keeps
    the complete canonical legend for static documentation and marker helpers.
    """
    styles = (*MPC_INTERACTION_STYLES, UNKNOWN_MPC_INTERACTION_STYLE)
    entries = tuple(
        MpcInteractionLegendEntry(
            interaction_type=style.interaction_type,
            label=style.label,
            compact_label=style.compact_label,
            color=_resolved_style_color(style, type_palette),
            tooltip=style.tooltip,
        )
        for style in styles
    )
    if present_types is None:
        return entries

    present = frozenset(int(value) for value in present_types)
    known_types = frozenset(
        style.interaction_type
        for style in MPC_INTERACTION_STYLES
        if style.interaction_type is not None
    )
    has_unknown = bool(present - known_types)
    return tuple(
        entry
        for entry in entries
        if (has_unknown if entry.interaction_type is None else entry.interaction_type in present)
    )


def rgb_to_css_hex(color: Sequence[float]) -> str:
    """Convert normalized RGB input to a stable CSS hex color."""
    rgb = np.asarray(color, dtype=np.float32).reshape(-1)
    if rgb.size < 3:
        rgb = np.asarray(MPC_UNKNOWN_COLOR, dtype=np.float32)
    values = np.clip(np.round(rgb[:3] * 255.0), 0.0, 255.0).astype(np.uint8)
    return f"#{int(values[0]):02x}{int(values[1]):02x}{int(values[2]):02x}"
