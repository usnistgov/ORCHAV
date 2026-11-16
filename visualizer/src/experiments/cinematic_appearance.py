"""Reuse canonical MPC color and recorded interaction semantics for presentation."""

import numpy as np

from ..metrics.mpc_canon import colorize
from ..renderers.pygfx.mpc import interaction_marker_spec

MAX_INTERACTION_LABELS = 32


def interaction_colors(catalog, point_indices, core, mode, *, distinct_materials=False):
    """Match bulk bounce RGB, including full-frame scalar limits/material fallback."""
    indices = np.asarray(point_indices, dtype=np.int64)
    neutral = np.full((len(indices), 3), 0.7, dtype=np.float32)
    if not len(indices):
        return neutral, "No recorded interactions"
    if core is None:
        return neutral, "MPC palette unavailable: neutral interaction colors"
    canon = catalog.canonical_data
    mask = np.zeros(len(canon.points), dtype=bool)
    mask[indices] = True
    if mode not in (
        "reflection_order",
        "mpc_type",
        "reconstruction_type",
        "delay",
        "path_loss",
        "material",
    ):
        return neutral, "MPC color mode unavailable: neutral interaction colors"
    materials = (
        core._resolve_material_colors(bool(distinct_materials)) if mode == "material" else None
    )
    colors = colorize(
        canon,
        mask,
        mode,
        core._order_palette,
        core._type_palette,
        core._viridis256,
        material_colors=materials,
    )
    # Mask output is in canonical vertex order; the animated subset may be ranked.
    mapped = colors[np.searchsorted(np.flatnonzero(mask), indices)]
    note = f"Interaction RGB: existing MPC {mode.replace('_', ' ')} colors"
    if mode == "material":
        note += "; per-vertex/outgoing material; missing material uses existing neutral fallback"
    return np.asarray(mapped, dtype=np.float32), note


def interaction_label_specs(geometry, *, enabled, show_types):
    """Bounded stable P/I labels; I is vertex ordinal, never inferred reflection count."""
    if not enabled:
        return ()
    indices = sorted(
        range(len(geometry.vertices)),
        key=lambda i: (
            int(geometry.interaction_path_ids[i]),
            int(geometry.interaction_ordinals[i]),
        ),
    )
    result = []
    for index in indices[:MAX_INTERACTION_LABELS]:
        path = int(geometry.interaction_path_ids[index])
        ordinal = int(geometry.interaction_ordinals[index])
        label = f"P{path} · I{ordinal}"
        if show_types:
            spec = interaction_marker_spec(int(geometry.interaction_types[index]))
            label += f" · {spec.label if spec is not None else 'Unknown'}"
        result.append((index, label))
    return tuple(result)
