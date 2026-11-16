"""Explain source-edit restrictions for derived visual surface partitions."""

from __future__ import annotations

from typing import Any


def visual_surface_source_edit_message(entry: dict[str, Any]) -> str | None:
    """Return guidance when one display fragment cannot edit its shared RF source.

    Surface partitions preserve one source shape's geometry and RF material.
    Editing an individual fragment would change only part of its display while
    serializing a change to the complete source shape. Visual appearance edits
    do not use this guard and remain available.
    """
    if not (entry.get("_visual_surface_partition") or entry.get("_geographic_layer_partition")):
        return None
    source = (
        entry.get("_visual_surface_source_name")
        or entry.get("_geographic_source_name")
        or entry.get("name")
        or "this mesh"
    )
    return (
        f"Geometry and RF material edits are unavailable for visual surfaces of '{source}'. "
        "Edit the source scene and its visual mapping, then reload. "
        "Visual appearance controls remain available."
    )
