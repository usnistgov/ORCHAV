"""Target-specific Sionna radio material construction."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from shared.logging import get_logger

from .defaults import (
    DEFAULT_TARGET_MATERIAL_THICKNESS,
    DEFAULT_TARGET_SCATTERING_COEFFICIENT,
)
from .overrides import MATERIAL_OVERRIDE_PROPS, set_material_property

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def available_target_material_types() -> tuple[str, ...]:
    """Return the exact ITU material keys accepted by the installed Sionna runtime."""

    from sionna.rt.radio_materials.itu import ITU_MATERIALS_PROPERTIES

    return tuple(sorted(str(name) for name in ITU_MATERIALS_PROPERTIES))


def validate_target_material_type(material_type: str) -> str:
    """Validate one target material against Sionna's construction registry."""

    value = str(material_type)
    available = available_target_material_types()
    if value not in available:
        choices = ", ".join(available)
        raise ValueError(f"Invalid ITU material type {value!r}; expected one of: {choices}")
    return value


def target_material_override_key_candidates(material_type: str) -> list[str]:
    """Return generic YAML override keys that can target a target material family."""
    return [
        material_type,
        f"itu_{material_type}",
        f"mat-itu_{material_type}",
    ]


def apply_target_material_overrides(
    target_material: Any,
    *,
    material_type: str,
    target_name: str,
    material_overrides: dict[str, dict[str, Any]] | None,
) -> None:
    """Apply generic material-family overrides to a generated target material."""
    if not material_overrides:
        return

    applied_keys: list[str] = []
    for key in target_material_override_key_candidates(material_type):
        props = material_overrides.get(key)
        if not props:
            continue
        for prop in MATERIAL_OVERRIDE_PROPS:
            if prop not in props or props[prop] is None:
                continue
            try:
                set_material_property(target_material, prop, props[prop])
                logger.info(
                    "Set target material %s.%s = %s from override '%s'",
                    getattr(target_material, "name", material_type),
                    prop,
                    props[prop],
                    key,
                )
            except (AttributeError, ValueError, TypeError) as exc:
                logger.warning(
                    "Failed to set target material override %s.%s = %s: %s",
                    key,
                    prop,
                    props[prop],
                    exc,
                )
        applied_keys.append(key)

    if applied_keys:
        logger.info(
            "Applied target material overrides for %s: %s",
            target_name,
            ", ".join(applied_keys),
        )


def create_target_material(
    *,
    material_type: str,
    target_name: str,
    material_overrides: dict[str, dict[str, Any]] | None = None,
) -> Any:
    """Create the per-target ITU material used by a Sionna ``SceneObject``."""
    material_type = validate_target_material_type(material_type)
    logger.info("Creating ITURadioMaterial for %s", material_type)
    from sionna.rt.radio_materials import ITURadioMaterial

    target_material = ITURadioMaterial(
        name=f"mat-itu_{material_type}_{target_name}",
        itu_type=material_type,
        thickness=DEFAULT_TARGET_MATERIAL_THICKNESS,
        color=None,
    )

    if hasattr(target_material, "scattering_coefficient"):
        target_material.scattering_coefficient = DEFAULT_TARGET_SCATTERING_COEFFICIENT
        logger.info(
            "Set target scattering_coefficient=%s for %s",
            DEFAULT_TARGET_SCATTERING_COEFFICIENT,
            target_name,
        )
    else:
        logger.warning("Target material doesn't support scattering_coefficient")

    apply_target_material_overrides(
        target_material,
        material_type=material_type,
        target_name=target_name,
        material_overrides=material_overrides,
    )

    logger.debug(
        "Created target material %s for %s",
        getattr(target_material, "name", material_type),
        target_name,
    )
    return target_material
