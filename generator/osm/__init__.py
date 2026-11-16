"""OpenStreetMap scene builder for Sionna RT.

Generates Mitsuba 3 XML scenes from OpenStreetMap building and water body data.
Contains code adapted from NVlabs/sionna-large-radio-maps (Apache 2.0).
See generator/osm/NOTICE.md for attribution details and list of changes.
"""

from __future__ import annotations

from .metadata import (
    OSM_METADATA_SCHEMA_VERSION,
    load_osm_metadata,
    normalize_osm_config,
    should_reuse_generated_scene,
)


def build_scene(*args, **kwargs):
    """Lazily import the heavy builder implementation on first use."""
    from .scene_builder import build_scene as _build_scene

    return _build_scene(*args, **kwargs)


def build_scene_from_center(*args, **kwargs):
    """Lazily import the heavy builder implementation on first use."""
    from .scene_builder import build_scene_from_center as _build_scene_from_center

    return _build_scene_from_center(*args, **kwargs)


__all__ = [
    "OSM_METADATA_SCHEMA_VERSION",
    "build_scene",
    "build_scene_from_center",
    "load_osm_metadata",
    "normalize_osm_config",
    "should_reuse_generated_scene",
]
