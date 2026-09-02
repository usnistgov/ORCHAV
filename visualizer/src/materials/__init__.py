"""Shared material helpers for canonical MPC data and renderer payloads.

Use ``catalog`` for visual material presets and PBR keyword conversion,
``texture_policy`` for launch-time texture-map rules, and
``canonical_materials`` for normalizing material labels used by canonical MPC
metadata.
"""

from .appearance import (
    AppearanceIntent,
    MaterialDisplayMode,
    MaterialGroupSummary,
    ResolvedAppearance,
    VisualMaterialBinding,
    VisualMaterialSource,
    resolve_appearance,
)
from .presets import BUILTIN_MATERIAL_PRESETS

__all__ = [
    "AppearanceIntent",
    "MaterialDisplayMode",
    "MaterialGroupSummary",
    "ResolvedAppearance",
    "VisualMaterialBinding",
    "VisualMaterialSource",
    "resolve_appearance",
    "BUILTIN_MATERIAL_PRESETS",
]
