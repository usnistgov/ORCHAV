"""Renderer-neutral exact grid boundaries with explicit available-data support."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from math import prod

import numpy as np

from ..model import RenderObject
from ..types.render_payloads import LineSetPayload, MaterialPayload
from .volume_cut import BoxCut

# Bound both Python edge bookkeeping and the eventual line payload. Dashed
# unavailable edges consume four segments each, included in this same budget.
MAX_CELL_EDGE_SEGMENTS = 100_000
_DASH_SEGMENTS = 4


def selected_cell_objects(axes, indices, *, normal=None, prefix="analysis:selected_cell"):
    """Outline one exact sampled cell as an on-top inspection annotation.

    Use a rectangle on the sample-centre plane for slices and a box otherwise.
    A dark border under the cyan stroke stays readable against the RF palette.
    These guides do not establish visibility or alter coverage classification.
    """
    axes = [np.asarray(axes[name]) for name in "xyz"]
    if len(indices) != 3 or any(not 0 <= i < len(a) for i, a in zip(indices, axes, strict=True)):
        raise ValueError("Selected cell is outside the recorded grid")
    axis = "xyz".index(normal) if normal is not None else None
    spacing = np.asarray([a[1] - a[0] for a in axes])
    origin = np.asarray([a[0] for a in axes]) - spacing / 2
    base = _object(
        "available",
        set(_cell_edges(indices, axis)),
        origin,
        spacing,
        prefix,
        slice_plane=axis is not None,
    )
    return tuple(
        replace(
            base,
            id=f"{prefix}:{name}",
            material=MaterialPayload(base_color=colour, line_width=width, shader="unlit"),
            metadata=dict(
                depth_test=False, depth_write=False, alpha_mode="blend", render_order=order
            ),
        )
        for name, colour, width, order in (
            ("border", (0.015, 0.025, 0.03, 1), 6.0, 32000),
            ("outline", (0.15, 1.0, 1.0, 1), 3.0, 32001),
        )
    )


def _cell_edges(cell, normal=None):
    """Yield integer-lattice edges; odd coordinates locate a slice centre."""
    dimensions = [axis for axis in range(3) if axis != normal]
    for direction in dimensions:
        others = [axis for axis in dimensions if axis != direction]
        for sides in product((0, 1), repeat=len(others)):
            start = [2 * index for index in cell]
            for axis, side in zip(others, sides, strict=True):
                start[axis] += 2 * side
            if normal is not None:
                start[normal] += 1
            yield (direction, *start)


def _object(kind, edges, origin, spacing, prefix, *, slice_plane):
    keys = np.asarray(sorted(edges), dtype=np.int64)
    start = origin + keys[:, 1:] * (spacing / 2)
    end = start.copy()
    end[np.arange(len(keys)), keys[:, 0]] += spacing[keys[:, 0]]
    if kind == "unavailable":
        # Four real line segments per edge give portable, renderer-neutral
        # dashes. Their endpoints preserve the exact cell boundary coordinates.
        fractions = np.linspace(0, 1, 2 * _DASH_SEGMENTS)
        points = start[:, None, :] + (end - start)[:, None, :] * fractions[None, :, None]
        color = (0x75 / 255, 0x75 / 255, 0x75 / 255, 1.0)
    else:
        points = np.stack((start, end), axis=1)
        color = (0x18 / 255, 0x24 / 255, 0x2B / 255, 1.0)
    points = points.reshape(-1, 3).astype(np.float32)
    return RenderObject(
        id=f"{prefix}:{kind}",
        payload=LineSetPayload(
            points=points,
            lines=np.arange(len(points), dtype=np.int32).reshape(-1, 2),
        ),
        material=MaterialPayload(base_color=color, line_width=1.5),
        metadata={
            "depth_compare": "<=",
            "depth_test": True,
            "depth_write": False,
            # Coplanar slice boundaries must follow the weighted volume pass.
            # Keep scene depth testing, so buildings still occlude the lines.
            # Explicit 3D defaults also restore an object reused after a slice.
            "alpha_mode": "blend" if slice_plane else "auto",
            "render_order": 10 if slice_plane else 0,
        },
    )


def grid_edge_objects(
    axes,
    available,
    *,
    enabled,
    include_unavailable=False,
    display="both",
    normal="z",
    index=0,
    cut=None,
    visible=True,
    prefix="analysis:cell_edges",
    available_label="Computed",
    unavailable_label="Unavailable",
    selected=None,
):
    """Return exact grid lines and a legend without changing sampled values.

    ``available`` is a boolean Z/Y/X grid. Borders ignore the displayed scalar
    quantity, its threshold and category filters. A slice draws rectangles on
    the sample-centre plane; a cut retains cells by their centre coordinates.
    The budget applies after deduplication, including dashed line segments.
    Oversized requests return an explanation, never a subsampled grid.
    """
    if not enabled or not visible:
        return [], ()
    axes = [np.asarray(axes[name]) for name in ("x", "y", "z")]
    available = np.asarray(available)
    if available.dtype != np.bool_ or available.shape != tuple(len(axis) for axis in axes[::-1]):
        raise ValueError("Cell-edge support must be a boolean Z/Y/X grid matching its axes")
    if selected is not None:
        selected = np.asarray(selected)
        if selected.shape != available.shape or selected.dtype != np.bool_:
            raise ValueError("Cell-edge selection must be a boolean mask matching availability")
    ranges = [range(len(axis)) for axis in axes]
    slice_normal = None
    if display == "slice":
        if normal not in "xyz" or len(normal) != 1:
            raise ValueError("Unknown cell-edge slice orientation")
        slice_normal = "xyz".index(normal)
        if index not in ranges[slice_normal]:
            raise ValueError("Cell-edge slice outside the sampled grid")
        ranges[slice_normal] = range(index, index + 1)
    elif isinstance(cut, BoxCut):
        retained = np.ones(available.shape, dtype=bool)
        retained[cut.slices(available.shape)] = False
        selected = retained if selected is None else np.asarray(selected) & retained
    elif cut is not None:
        axis, boundary, keep_lower = cut
        if axis not in "xyz" or len(axis) != 1:
            raise ValueError("Unknown cell-edge cut orientation")
        dimension = "xyz".index(axis)
        if boundary not in ranges[dimension]:
            raise ValueError("Cell-edge cut outside the sampled grid")
        ranges[dimension] = (
            range(boundary + 1) if keep_lower else range(boundary, len(axes[dimension]))
        )
    selection = tuple(slice(r.start, r.stop) for r in ranges[::-1])
    if selected is not None:
        available_count = int(np.count_nonzero(available[selection] & selected[selection]))
        unavailable_count = int(np.count_nonzero(~available[selection] & selected[selection]))
    else:
        available_count = int(np.count_nonzero(available[selection]))
        unavailable_count = prod(map(len, ranges)) - available_count
    legend = [
        f"Solid dark edges: {available_count} {available_label.lower()} cells",
        (
            "Cell edges follow the selected display region"
            if selected is not None
            else "Cell edges are independent of fills / thresholds"
        ),
    ]
    if include_unavailable:
        legend.append(f"Dashed gray: {unavailable_count} {unavailable_label.lower()} cells")
    else:
        legend.append(f"{unavailable_label} cells omitted: {unavailable_count}")
    groups = {"available": set(), "unavailable": set()}
    segments = 0
    for kind in ("available", "unavailable"):
        if kind == "unavailable" and not include_unavailable:
            continue
        wanted = kind == "available"
        if (available_count if wanted else unavailable_count) == 0:
            continue
        cells = (
            cell
            for cell in product(*ranges)
            if bool(available[cell[::-1]]) == wanted and (selected is None or selected[cell[::-1]])
        )
        weight = _DASH_SEGMENTS if kind == "unavailable" else 1
        for cell in cells:
            for edge in _cell_edges(cell, slice_normal):
                if edge in groups[kind]:
                    continue
                segments += weight
                if segments > MAX_CELL_EDGE_SEGMENTS:
                    return [], (
                        f"Cell edges disabled: more than {MAX_CELL_EDGE_SEGMENTS:,} line segments",
                        "Choose a slice or narrower cutaway; no cells were subsampled",
                    )
                groups[kind].add(edge)
    spacing = np.asarray([axis[1] - axis[0] for axis in axes])
    origin = np.asarray([axis[0] for axis in axes]) - spacing / 2
    # Shared borders do not change either cell's availability.
    objects = [
        _object(kind, groups[kind], origin, spacing, prefix, slice_plane=slice_normal is not None)
        for kind in ("unavailable", "available")
        if groups[kind]
    ]
    return objects, tuple(legend)
