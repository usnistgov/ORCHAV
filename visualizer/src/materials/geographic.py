"""Compatibility import for generated-map surface display defaults.

The material resolver also uses these defaults for prepared scenes in the
non-map export. Keep that implementation in the common display layer.
"""

from .surface_defaults import geographic_surface_properties

__all__ = ["geographic_surface_properties"]
