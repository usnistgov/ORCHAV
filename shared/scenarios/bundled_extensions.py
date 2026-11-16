"""Register the reviewed public geographic-scene producer without private hooks."""

from __future__ import annotations


def register() -> None:
    """Expose OSM scene recipes; downloading still requires an explicit build."""
    from generator.osm.scenario_extension import register as register_osm

    register_osm()
