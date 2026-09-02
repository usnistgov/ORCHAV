"""Apply configured radio material policy to loaded Sionna RT scenes.

This module is the package entry point for scene setup. It can assign ORCHAV's
known scattering coefficients to Sionna-loaded scene materials by material
family, then apply explicit per-material YAML overrides. Explicit overrides are
last so scenario values always win over broad defaults.
"""

from __future__ import annotations

from typing import Any

from shared.logging import get_logger

from .defaults import DEFAULT_SCATTERING_COEFFICIENTS
from .overrides import apply_material_overrides

logger = get_logger(__name__)

# Public YAML presets for
# ``raytracing.scene_materials.scattering_coefficient_preset``.
SCENE_MATERIAL_SCATTERING_PRESETS: tuple[str, ...] = ("none", "itu")


def resolve_scene_material_family(
    material_name: str,
    *,
    known_families: set[str] | None = None,
) -> str | None:
    """Resolve a loaded scene material name to a known ORCHAV material family.

    Matches exact material-family names and common Sionna XML identifiers such
    as ``itu_concrete`` and ``mat-itu_concrete``. Names with additional suffixes
    such as generated target materials (``mat-itu_glass_Target``) are skipped.
    """
    if known_families is None:
        known_families = set(DEFAULT_SCATTERING_COEFFICIENTS)

    normalized = str(material_name).strip().lower()
    if not normalized:
        return None

    candidates = [normalized]
    for separator in ("-", "/"):
        if separator in normalized:
            candidates.append(normalized.rsplit(separator, maxsplit=1)[-1])

    for candidate in candidates:
        if candidate.startswith("mat-"):
            candidate = candidate[len("mat-") :]
        if candidate.startswith("itu_"):
            candidate = candidate[len("itu_") :]
        if candidate in known_families:
            return candidate
    return None


def _material_name_candidates(key: Any, material: Any) -> list[str]:
    """Return scene dictionary and material-object names without duplicates."""
    candidates: list[str] = []
    for value in (key, getattr(material, "name", None)):
        if value is None:
            continue
        name = str(value)
        if name and name not in candidates:
            candidates.append(name)
    return candidates


def _apply_default_scene_scattering(
    scene: Any,
    scattering_coefficients: dict[str, float],
) -> None:
    """Apply material-family scattering coefficients to loaded scene materials."""
    logger.info("Applying default scattering coefficients to loaded scene materials...")
    for key, material in getattr(scene, "radio_materials", {}).items():
        try:
            family = None
            for name in _material_name_candidates(key, material):
                family = resolve_scene_material_family(
                    name,
                    known_families=set(scattering_coefficients),
                )
                if family is not None:
                    break
            if family is None:
                logger.debug("%s: no known material family; skipping", key)
                continue
            if not hasattr(material, "scattering_coefficient"):
                logger.debug("%s: material has no scattering_coefficient; skipping", key)
                continue
            material.scattering_coefficient = scattering_coefficients[family]
            logger.info(
                "%s: %s with scattering_coefficient=%s",
                key,
                family,
                scattering_coefficients[family],
            )
        except (AttributeError, ValueError, TypeError) as exc:
            logger.debug("Could not set scattering coefficient for '%s': %s", key, exc)
    logger.info("Default scattering coefficients applied to loaded scene materials")


def apply_material_settings(
    scene: Any,
    *,
    default_scattering_preset: str = "none",
    scattering_coefficients: dict[str, float] | None = None,
    material_overrides: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Apply optional scene-material defaults and explicit material overrides.

    ``default_scattering_preset='itu'`` assigns ORCHAV's known scattering
    coefficients to loaded scene materials by material-family name. ``'none'``
    leaves all Sionna-loaded scene material values untouched. Explicit material
    overrides are applied last whenever provided, so user YAML values are
    authoritative.
    """
    # YAML stores this as a preset name, not a numeric coefficient. Normalize
    # here so values such as " ITU " behave like "itu", then reject typos.
    preset = str(default_scattering_preset or "none").strip().lower()
    if preset not in SCENE_MATERIAL_SCATTERING_PRESETS:
        raise ValueError(
            "scattering_coefficient_preset must be one of "
            f"{SCENE_MATERIAL_SCATTERING_PRESETS}, got {default_scattering_preset!r}"
        )

    if preset == "itu":
        _apply_default_scene_scattering(
            scene,
            scattering_coefficients or DEFAULT_SCATTERING_COEFFICIENTS,
        )

    if material_overrides:
        apply_material_overrides(scene, material_overrides)
