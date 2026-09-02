"""Apply explicit YAML material overrides to Sionna RT materials.

``settings.py`` calls this module after any broad scene-material defaults. The
override keys are scenario-facing material names; this module resolves those
aliases to Sionna scene material keys, coerces string scattering-pattern names
through Sionna's registry, and routes ``alpha_r``/``alpha_g`` to the material or
its scattering-pattern object.
"""

from __future__ import annotations

import math
from typing import Any

from shared.logging import get_logger

logger = get_logger(__name__)

MATERIAL_OVERRIDE_PROPS: tuple[str, ...] = (
    "relative_permittivity",
    "conductivity",
    "scattering_coefficient",
    "scattering_pattern",
    "alpha_r",
    "alpha_g",
    "xpd_coefficient",
    "thickness",
)


def _scattering_pattern_name(value: Any) -> str | None:
    """Return a normalized pattern name from a string or pattern-like object."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower()
    name = getattr(value, "name", None)
    if isinstance(name, str):
        normalized = name.strip().lower()
        if normalized.endswith("pattern"):
            normalized = normalized[: -len("pattern")]
        return normalized
    cls_name = type(value).__name__.strip().lower()
    if cls_name.endswith("pattern"):
        cls_name = cls_name[: -len("pattern")]
    return cls_name or None


def _canonical_scattering_pattern_name(value: Any) -> str | None:
    """Map user-friendly scattering-pattern aliases to registered names."""
    name = _scattering_pattern_name(value)
    if name is None:
        return None
    aliases = {
        "gaussianrer": "g-rer",
        "gaussian_rer": "g-rer",
        "grer": "g-rer",
        "iso": "isotropic",
        "erisotropic": "er-isotropic",
        "er_isotropic": "er-isotropic",
    }
    return aliases.get(name, name)


def _register_custom_scattering_patterns() -> None:
    """Ensure ORCHAV-supported scattering-pattern names exist in Sionna's registry."""
    from .scattering_patterns import register_extra_scattering_patterns

    register_extra_scattering_patterns()


def _get_scattering_pattern_factory(pattern_name: str):
    """Return Sionna's registered factory for a scattering-pattern name."""
    from sionna.rt.radio_materials.scattering_pattern import scattering_pattern_registry

    factory = scattering_pattern_registry.get(pattern_name)
    if factory is None:
        raise ValueError(f"Unknown scattering_pattern '{pattern_name}'")
    return factory


def _coerce_scattering_pattern(value: Any):
    """Instantiate string scattering-pattern overrides and pass objects through."""
    if not isinstance(value, str):
        return value
    pattern_name = _canonical_scattering_pattern_name(value)
    if not pattern_name:
        raise ValueError("Empty scattering_pattern is not allowed")
    _register_custom_scattering_patterns()
    factory = _get_scattering_pattern_factory(pattern_name)
    return factory()


def _maybe_instantiate_scattering_pattern(scattering: Any, required_attr: str) -> Any:
    """Instantiate a named pattern only if it can hold the requested alpha field."""
    if not isinstance(scattering, str):
        return scattering
    candidate = _coerce_scattering_pattern(scattering)
    if hasattr(candidate, required_attr):
        return candidate
    return scattering


def _is_alpha_r_capable(value: Any) -> bool:
    """Return whether a material or named pattern supports ``alpha_r``."""
    if value is None:
        return False
    if hasattr(value, "alpha_r"):
        return True
    return _canonical_scattering_pattern_name(value) in {"directive", "backscattering"}


def _is_alpha_g_capable(value: Any) -> bool:
    """Return whether a material or named pattern supports ``alpha_g``."""
    if value is None:
        return False
    if hasattr(value, "alpha_g"):
        return True
    return _canonical_scattering_pattern_name(value) == "g-rer"


def _coerce_alpha_r(value: Any) -> int:
    """Coerce ``alpha_r`` to Sionna's positive integer directive/backscatter order."""
    alpha = float(value)
    if not math.isfinite(alpha):
        raise ValueError(f"alpha_r must be finite, got {alpha!r}")
    return max(1, int(round(alpha)))


def _coerce_alpha_g(value: Any) -> float:
    """Coerce ``alpha_g`` to a finite Gaussian RER lobe parameter."""
    alpha = float(value)
    if not math.isfinite(alpha):
        raise ValueError(f"alpha_g must be finite, got {alpha!r}")
    return alpha


def set_material_property(material: Any, prop: str, value: Any) -> None:
    """Set one supported override property on a Sionna RT material.

    String scattering-pattern values are instantiated through Sionna's registry.
    ``alpha_r`` and ``alpha_g`` are applied to the material itself when possible,
    otherwise to the material's scattering-pattern object.
    """
    if prop == "scattering_pattern":
        setattr(material, prop, _coerce_scattering_pattern(value))
        return

    if prop == "alpha_r":
        alpha_int = _coerce_alpha_r(value)
        # Some Sionna material classes expose alpha fields directly; others keep
        # them on the scattering-pattern object.
        if hasattr(material, "alpha_r"):
            try:
                setattr(material, "alpha_r", alpha_int)
                return
            except (AttributeError, TypeError, ValueError):
                pass
        scattering = getattr(material, "scattering_pattern", None)
        # A configured pattern may still be stored as a string. Instantiate it
        # only when the registered pattern supports the requested alpha field.
        if isinstance(scattering, str) and _is_alpha_r_capable(scattering):
            candidate = _maybe_instantiate_scattering_pattern(scattering, "alpha_r")
            if hasattr(candidate, "alpha_r"):
                setattr(material, "scattering_pattern", candidate)
                scattering = candidate
        if scattering is not None and hasattr(scattering, "alpha_r"):
            setattr(scattering, "alpha_r", alpha_int)
            return
        raise AttributeError(
            f"{type(material).__name__} has no alpha-capable scattering pattern for 'alpha_r'"
        )

    if prop == "alpha_g":
        alpha_float = _coerce_alpha_g(value)
        # Prefer direct material fields, but support Sionna patterns that carry
        # Gaussian RER parameters on the nested pattern object.
        if hasattr(material, "alpha_g"):
            try:
                setattr(material, "alpha_g", alpha_float)
                return
            except (AttributeError, TypeError, ValueError):
                pass
        scattering = getattr(material, "scattering_pattern", None)
        if isinstance(scattering, str) and _is_alpha_g_capable(scattering):
            candidate = _maybe_instantiate_scattering_pattern(scattering, "alpha_g")
            if hasattr(candidate, "alpha_g"):
                setattr(material, "scattering_pattern", candidate)
                scattering = candidate
        if scattering is not None and hasattr(scattering, "alpha_g"):
            setattr(scattering, "alpha_g", alpha_float)
            return
        raise AttributeError(
            f"{type(material).__name__} has no alpha-capable scattering pattern for 'alpha_g'"
        )

    setattr(material, prop, value)


def resolve_material_name(scene: Any, requested_name: str) -> str | None:
    """Resolve a configured material name to a Sionna scene material key.

    Accepts bare material names and common Sionna prefixes such as ``itu_``,
    ``mat-``, and ``mat-itu_``.
    """
    materials = scene.radio_materials
    if requested_name in materials:
        return requested_name

    candidates = [requested_name]
    if requested_name.startswith("mat-"):
        candidates.append(requested_name[4:])
    if requested_name.startswith("mat-itu_"):
        suffix = requested_name[len("mat-itu_") :]
        candidates.extend([f"itu_{suffix}", suffix])
    if requested_name.startswith("mat-") and not requested_name.startswith("mat-itu_"):
        suffix = requested_name[len("mat-") :]
        candidates.append(f"mat-itu_{suffix}")
    if requested_name.startswith("itu_"):
        suffix = requested_name[len("itu_") :]
        candidates.extend([f"mat-itu_{suffix}", f"mat-{requested_name}", suffix])
    if not requested_name.startswith("mat-"):
        candidates.append(f"mat-{requested_name}")
        candidates.append(f"mat-itu_{requested_name}")

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in materials:
            logger.debug(
                "Material alias '%s' resolved to scene material '%s'",
                requested_name,
                candidate,
            )
            return candidate
    return None


def apply_material_overrides(
    scene: Any,
    material_overrides: dict[str, dict[str, Any]],
) -> None:
    """Apply explicit ``raytracing.materials`` overrides to scene materials.

    Overrides may set material parameters, scattering patterns, or
    scattering-pattern alpha parameters. Missing materials and invalid property
    values are logged and skipped so one typo does not abort scene setup.
    """
    for mat_name, props in material_overrides.items():
        resolved = resolve_material_name(scene, mat_name)
        if resolved is None:
            logger.warning(
                "Material override '%s' not found in scene; skipping. Available: %s",
                mat_name,
                list(scene.radio_materials.keys())[:10],
            )
            continue

        material = scene.radio_materials[resolved]
        for prop in MATERIAL_OVERRIDE_PROPS:
            value = props.get(prop)
            if value is None:
                continue
            try:
                set_material_property(material, prop, value)
                logger.info("Set %s.%s = %s", resolved, prop, value)
            except (AttributeError, ValueError, TypeError) as exc:
                logger.warning("Failed to set %s.%s = %s: %s", resolved, prop, value, exc)
