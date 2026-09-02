"""Built-in material-family defaults for optional scene scattering.

These values are not assigned automatically. ``settings.py`` applies them only
when ``raytracing.scene_materials.scattering_coefficient_preset`` is ``"itu"``.
Explicit ``raytracing.materials`` overrides are applied later and remain
authoritative.
"""

from __future__ import annotations

# Scene-specific names such as ``itu_concrete`` and ``mat-itu_concrete`` are
# resolved to these family keys before the optional preset is applied. Sionna's
# coefficient convention is 0 = pure specular and 1 = pure diffuse.
DEFAULT_SCATTERING_COEFFICIENTS: dict[str, float] = {
    "marble": 0.08,
    "glass": 0.05,
    "metal": 0.10,
    "wood": 0.25,
    "plywood": 0.20,
    "chipboard": 0.25,
    "plasterboard": 0.35,
    "concrete": 0.40,
    "ceiling_board": 0.50,
}

# Target material defaults are used by target mesh setup, not by the loaded scene
# material preset above.
DEFAULT_TARGET_MATERIAL_THICKNESS = 0.1
DEFAULT_TARGET_SCATTERING_COEFFICIENT = 0.3
