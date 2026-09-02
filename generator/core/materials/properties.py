"""Export resolved scalar RF material properties from a Sionna RT scene."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

RF_MATERIAL_PROPERTY_FIELDS: tuple[str, ...] = (
    "relative_permittivity",
    "conductivity",
    "scattering_coefficient",
    "xpd_coefficient",
    "thickness",
)


def collect_scene_radio_material_properties(scene: Any) -> dict[str, Any]:
    """Return JSON-ready RF material properties resolved on a Sionna scene.

    Sionna material attributes may be Python scalars, tensors, or DrJit-like
    values. This helper only exports finite scalar fields and leaves unknown or
    vector-valued fields out of the manifest.
    """
    properties: dict[str, dict[str, float]] = {}
    materials = getattr(scene, "radio_materials", None)
    if not isinstance(materials, Mapping):
        return {
            "schema_version": 1,
            "source": "sionna.rt.Scene.radio_materials",
            "properties": properties,
        }

    for raw_name, material in materials.items():
        material_names = _material_name_candidates(raw_name, material)
        scalars = _material_to_scalars(material)
        if not scalars:
            continue
        for name in material_names:
            properties[name] = dict(scalars)

    return {
        "schema_version": 1,
        "source": "sionna.rt.Scene.radio_materials",
        "properties": properties,
    }


def _material_name_candidates(raw_name: Any, material: Any) -> tuple[str, ...]:
    """Return stable names by which consumers may match a material."""
    names: list[str] = []
    for value in (raw_name, getattr(material, "name", None)):
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in names:
            names.append(text)
    return tuple(names)


def _material_to_scalars(material: Any) -> dict[str, float]:
    """Extract supported finite scalar RF properties from one material."""
    values: dict[str, float] = {}
    for field_name in RF_MATERIAL_PROPERTY_FIELDS:
        try:
            raw_value = getattr(material, field_name)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            continue
        scalar = _coerce_scalar(raw_value)
        if scalar is not None:
            values[field_name] = scalar
    return values


def _coerce_scalar(value: Any) -> float | None:
    """Coerce scalar-like tensor/array/Python values to a finite float."""
    if value is None or callable(value):
        return None

    for attr_name in ("numpy", "item"):
        method = getattr(value, attr_name, None)
        if callable(method):
            try:
                value = method()
            except (TypeError, ValueError, RuntimeError):
                return None

    try:
        scalar = float(value)
    except (TypeError, ValueError):
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return None
        if array.size != 1:
            return None
        try:
            scalar = float(array.reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            return None

    return scalar if np.isfinite(scalar) else None
