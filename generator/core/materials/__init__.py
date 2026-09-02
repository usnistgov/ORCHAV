"""Generator-side helpers for Sionna RT radio materials.

This package handles the material policy that is applied after a Sionna scene is
loaded: optional ORCHAV defaults for scene-material scattering coefficients,
explicit YAML material overrides, and registration of extra scattering-pattern
names accepted by those overrides.

Start with :func:`apply_material_settings` in ``settings.py`` for the normal
scene setup path. ``overrides.py`` applies per-material electromagnetic
properties, ``defaults.py`` holds ORCHAV material-family defaults,
``scattering_patterns.py`` registers string pattern names with Sionna,
``target_materials.py`` constructs generated target materials, and
``path_metadata.py`` extracts per-bounce material labels from solved paths.
"""

from .overrides import (
    MATERIAL_OVERRIDE_PROPS,
    apply_material_overrides,
    resolve_material_name,
    set_material_property,
)
from .properties import RF_MATERIAL_PROPERTY_FIELDS, collect_scene_radio_material_properties
from .scattering_patterns import register_extra_scattering_patterns
from .settings import apply_material_settings, resolve_scene_material_family
from .target_materials import (
    apply_target_material_overrides,
    create_target_material,
    target_material_override_key_candidates,
)

__all__ = [
    "MATERIAL_OVERRIDE_PROPS",
    "RF_MATERIAL_PROPERTY_FIELDS",
    "apply_target_material_overrides",
    "apply_material_overrides",
    "apply_material_settings",
    "collect_scene_radio_material_properties",
    "create_target_material",
    "register_extra_scattering_patterns",
    "resolve_material_name",
    "resolve_scene_material_family",
    "set_material_property",
    "target_material_override_key_candidates",
]
