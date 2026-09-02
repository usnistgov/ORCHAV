"""Renderer-neutral target material policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..materials.catalog import is_known_material_type, material_preset

DEFAULT_TARGET_PBR_PROPS: dict[str, Any] = {
    "color": [0.8, 0.6, 0.5],
    "roughness": 0.6,
    "metallic": 0.0,
    "reflectance": 0.4,
}
MINIMAL_TARGET_PBR_PROPS: dict[str, Any] = {
    "color": [0.8, 0.6, 0.5],
    "roughness": 0.6,
    "metallic": 0.0,
}
TARGET_MATERIAL_FALLBACKS: dict[str, dict[str, Any]] = {
    "concrete": {
        "color": [0.7, 0.7, 0.7],
        "roughness": 0.8,
        "metallic": 0.0,
        "reflectance": 0.3,
    },
    "glass": {
        "color": [0.85, 0.92, 0.98],
        "roughness": 0.1,
        "metallic": 0.0,
        "reflectance": 0.9,
    },
    "metal": {
        "color": [0.9, 0.9, 0.9],
        "roughness": 0.6,
        "metallic": 0.7,
        "reflectance": 0.4,
    },
}


@dataclass(frozen=True)
class TargetPbrResolution:
    """Resolved PBR properties for one target material."""

    props: dict[str, Any]
    visual_profile_matched: bool = False


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: list(item) if isinstance(item, list) else item for key, item in value.items()}


def resolve_target_pbr_props(
    *,
    target_name: str,
    material_type: str,
    visual_profile_service: Any = None,
    default_props: Mapping[str, Any] = DEFAULT_TARGET_PBR_PROPS,
    use_material_fallbacks: bool = True,
) -> TargetPbrResolution:
    """Resolve target PBR props from profile, catalog, then stable fallback values."""
    profile_props = None
    resolve = getattr(visual_profile_service, "resolve", None)
    if callable(resolve):
        profile_props = resolve(target_name, material_type, "target")
    if isinstance(profile_props, Mapping):
        return TargetPbrResolution(_copy_mapping(profile_props), visual_profile_matched=True)

    key = str(material_type or "").strip().lower()
    catalog_props = material_preset(key)
    if (
        use_material_fallbacks
        and is_known_material_type(key)
        and key != "default"
        and isinstance(catalog_props, Mapping)
    ):
        return TargetPbrResolution(_copy_mapping(catalog_props))
    if use_material_fallbacks and key in TARGET_MATERIAL_FALLBACKS:
        return TargetPbrResolution(_copy_mapping(TARGET_MATERIAL_FALLBACKS[key]))
    return TargetPbrResolution(_copy_mapping(default_props))


def target_handle_color(props: Mapping[str, Any], *, has_vertex_texture: bool) -> list[float]:
    """Return the material base color used while uploading a target handle."""
    if has_vertex_texture:
        return [1.0, 1.0, 1.0]
    color = props.get("color", DEFAULT_TARGET_PBR_PROPS["color"])
    return list(color[:3]) if isinstance(color, list) else list(color)


def target_entry_pbr_fields(props: Mapping[str, Any]) -> dict[str, Any]:
    """Return target-entry PBR fields derived from resolved material props."""
    color = props.get("color", DEFAULT_TARGET_PBR_PROPS["color"])
    color_list = list(color[:3]) if isinstance(color, list) else list(color)
    complete_props = _copy_mapping(props)
    complete_props["color"] = color_list
    complete_props.setdefault("alpha", 1.0)
    return {
        "pbr_color": color_list,
        "pbr_roughness": props.get("roughness", 0.5),
        "pbr_metallic": props.get("metallic", 0.0),
        "pbr_reflectance": props.get("reflectance", 0.4),
        # Retain the complete visual profile, including advanced scalar and
        # texture-map fields, so later highlight/material refreshes restore
        # exactly the material used for the initial target frame.
        "pbr_properties": complete_props,
    }


def target_entry_base_pbr_props(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return the base material properties stored on a target entry."""
    return {
        "color": entry.get("color", [0.7, 0.7, 0.7]),
        "roughness": entry.get("pbr_roughness", 0.5),
        "metallic": entry.get("pbr_metallic", 0.0),
        "reflectance": entry.get("pbr_reflectance", 0.5),
        "alpha": entry.get("pbr_alpha", 1.0),
    }


def resolve_effective_target_entry_pbr_props(
    entry: Mapping[str, Any],
    material_pbr_service: Any = None,
) -> dict[str, Any]:
    """Apply material-service overrides to a target entry's base PBR props."""
    base_props = target_entry_base_pbr_props(entry)
    if material_pbr_service is None:
        return base_props

    get_entry = getattr(material_pbr_service, "get_effective_entry_properties", None)
    if callable(get_entry):
        resolved = get_entry(entry)
        if isinstance(resolved, Mapping):
            return _copy_mapping(resolved)

    get_material = getattr(material_pbr_service, "get_effective_properties", None)
    if callable(get_material):
        material_type = str(entry.get("material_type") or "default")
        resolved = get_material(material_type, base_props)
        if isinstance(resolved, Mapping):
            return _copy_mapping(resolved)

    return base_props
