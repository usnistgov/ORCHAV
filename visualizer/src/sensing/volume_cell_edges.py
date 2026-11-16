"""Sensing sampling support independent of outcome and native-file availability."""

from __future__ import annotations

import numpy as np

from ..utils.cell_edges import grid_edge_objects


def cell_edge_objects(
    reader,
    heading,
    *,
    enabled,
    include_unavailable=False,
    display="both",
    normal="z",
    index=0,
    cut=None,
    visible=True,
):
    """Outline measured trials, optionally distinguishing excluded/unknown cells.

    A measured trial with zero reports or omitted native files remains measured.
    Missing, pending and excluded records have no acquired outcome; they are
    never drawn as missed detections. Fill filters do not hide sampling support.
    """
    if not enabled or not visible:
        return [], ()
    support = np.zeros(tuple(len(reader.axes[name]) for name in ("z", "y", "x")), bool)
    for (h, z, y, x), trial in reader.cells.items():
        if h == heading and trial["status"] == "measured":
            support[z, y, x] = True
    objects, legend = grid_edge_objects(
        reader.axes,
        support,
        enabled=enabled,
        include_unavailable=include_unavailable,
        display=display,
        normal=normal,
        index=index,
        cut=cut,
        visible=visible,
        prefix="sensing:volume:cell_edges",
        available_label="Measured",
        unavailable_label="Excluded / unmeasured",
    )
    if objects:
        legend += (
            "Measured includes no-report trials; dashed cells have no acquired outcome",
            "Edges remain when outcome/focus filters hide fills",
        )
    return objects, legend
