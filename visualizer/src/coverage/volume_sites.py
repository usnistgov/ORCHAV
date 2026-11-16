"""Recorded transmitter locators for native coverage-volume inspection."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from shared.coverage.volume import ALL_TRANSMITTERS

from ..services.site_locator_service import site_locator_objects

SITE_OBJECT_IDS = (
    "analysis:coverage_transmitter",
    "analysis:coverage_transmitter_locator",
    "analysis:coverage_transmitter_label",
)


def transmitter_position(source, name):
    """Return one unambiguous recorded XYZ position, never an inferred origin."""
    names = source.get("tx_names", [])
    if list(names).count(name) != 1:
        return None
    try:
        positions = np.asarray(source.get("tx_positions"), dtype=float)
    except (TypeError, ValueError):
        return None
    if positions.shape != (len(names), 3):
        return None
    position = positions[list(names).index(name)]
    return position.copy() if np.isfinite(position).all() else None


def transmitter_objects(source, name, axes, *, readable=False):
    """Place a dot at the transmitter and a labelled locator above the volume.

    The vertical line locates the recorded point through opaque coverage cells;
    it is an interface annotation, not an antenna mast or propagation path.
    """
    if name == ALL_TRANSMITTERS:
        return [
            replace(
                obj,
                id=f"{obj.id}:{index}",
                metadata={
                    **obj.metadata,
                    "locator_companions": [
                        f"{name}:{index}" for name in obj.metadata.get("locator_companions", ())
                    ],
                },
            )
            for index, tx in enumerate(source.get("tx_names", []))
            for obj in transmitter_objects(source, tx, axes, readable=readable)
        ]
    position = transmitter_position(source, name)
    if position is None:
        return []
    display_name = source.get("tx_display_names", {}).get(name, f"TX {name}")
    return site_locator_objects(position, display_name, axes, SITE_OBJECT_IDS, readable=readable)


def transmitter_object_ids(source):
    """Include individual and combined locators for source-switch cleanup."""
    return (
        *SITE_OBJECT_IDS,
        *(
            f"{object_id}:{index}"
            for index in range(len(source.get("tx_names", [])))
            for object_id in SITE_OBJECT_IDS
        ),
    )
