"""Renderer-neutral material presets and PBR payload helpers."""

from __future__ import annotations

import os as _os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional

from shared.logging import get_logger
from shared.scenarios.paths import find_project_root

from ..types.render_payloads import MaterialPayload, material_payload_from_mapping
from .texture_policy import (
    TEXTURE_MAP_KEYS,
    TEXTURE_POLICY_RESOLVED_KEY,
    TexturePolicyResult,
    apply_texture_policy_to_props,
    textures_globally_enabled,
    warn_for_texture_policy,
)

logger = get_logger("orchav.materials.catalog")


class _FrozenList(tuple):
    """Tuple marker that lets detached copies restore authored lists."""


def _freeze_material_value(value: Any) -> Any:
    """Recursively detach mutable material values for a shared resolution."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_material_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return _FrozenList(_freeze_material_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_material_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_material_value(item) for item in value)
    return value


def _thaw_material_value(value: Any) -> Any:
    """Return a detached mutable value with original sequence shapes."""
    if isinstance(value, Mapping):
        return {key: _thaw_material_value(item) for key, item in value.items()}
    if isinstance(value, _FrozenList):
        return [_thaw_material_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_thaw_material_value(item) for item in value)
    if isinstance(value, frozenset):
        return {_thaw_material_value(item) for item in value}
    return value


@dataclass(frozen=True, slots=True)
class ResolvedMaterial:
    """Immutable application-boundary material and texture-policy result."""

    properties: Mapping[str, Any]
    texture_policy: TexturePolicyResult
    payload: MaterialPayload

    def __post_init__(self) -> None:
        """Detach authored property containers before sharing this result."""
        object.__setattr__(self, "properties", _freeze_material_value(self.properties))

    def properties_copy(self, *, mark_texture_policy: bool = False) -> dict[str, Any]:
        """Return mutable properties without repeating path validation."""
        props = _thaw_material_value(self.properties)
        if mark_texture_policy:
            props[TEXTURE_POLICY_RESOLVED_KEY] = True
        return props


# Curated Poly Haven CC0 starter pack under libraries/textures/pbr/. The
# presets below reference absolute paths resolved once at import time so
# renderer texture caches can short-circuit identity lookups. See
# libraries/textures/pbr/CREDITS.md for provenance.
try:
    _PBR_MAPS_ROOT = find_project_root(Path(__file__).parent) / "libraries" / "textures" / "pbr"
except (OSError, ValueError):
    _PBR_MAPS_ROOT = Path("libraries/textures/pbr")

# Texture paths stay in the catalog; the shared launch policy decides whether
# either renderer may bind them.
_SSR_ENABLED: bool = _os.environ.get("ORCHAV_ENABLE_SSR") == "1"
_MATERIAL_TYPE_PREFIXES = ("mat-itu_", "mat_itu_", "itu_", "itu-", "mat-")
_MATERIAL_ID_PREFIXES = ("mat-itu_", "mat_itu_", "mat-", "itu_", "itu-")
_MATERIAL_ID_SUFFIXES = ("_mat", "_material")


def textures_disabled() -> bool:
    """Return True when visualizer launch opts out of all texture maps."""
    return not textures_globally_enabled()


def _pbr_map(slot: str, map_name: str) -> Optional[str]:
    """Return the absolute path to a bundled PBR map, or None."""
    path = _PBR_MAPS_ROOT / slot / f"{map_name}.png"
    return str(path) if path.exists() else None


def _ssr(variant: str = "defaultLitSSR") -> Optional[str]:
    """Return the optional Open3D SSR shader variant."""
    return variant if _SSR_ENABLED else None


def normalize_material_type_name(material_type: object, default: str = "default") -> str:
    """Normalize scene and UI material names to catalog lookup keys."""
    if not material_type:
        return default
    name = str(material_type).strip().lower()
    for prefix in _MATERIAL_TYPE_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name or default


# Maps ITU material type strings to renderer-neutral PBR properties.
ITU_TO_PBR: dict[str, dict[str, object]] = {
    # Base colors aligned with Sionna RT's ITURadioMaterial.ITU_MATERIAL_COLORS
    # so that the visualizer matches Sionna's scene preview appearance.
    "concrete": {
        "color": [0.539, 0.539, 0.539],
        "roughness": 0.8,
        "metallic": 0.0,
        "reflectance": 0.3,
        "alpha": 1.0,
        "texture_path": _pbr_map("concrete", "albedo"),
        "normal_map_path": _pbr_map("concrete", "normal"),
        "normal_map_strength": 1.35,
        "roughness_map_path": _pbr_map("concrete", "roughness"),
        "ao_map_path": _pbr_map("concrete", "ao"),
        "uv_scale_meters": 2.5,
    },
    "glass": {
        "color": [0.168, 0.139, 0.509],
        "roughness": 0.05,
        "metallic": 0.0,
        "reflectance": 0.95,
        "alpha": 0.7,
        "transmission": 0.9,
        "glass_thickness": 0.5,
        "absorption_color": [0.92, 0.95, 1.0],
        "shader_variant": _ssr(),
    },
    "metal": {
        "color": [0.220, 0.220, 0.254],
        "roughness": 0.6,
        "metallic": 0.7,
        "reflectance": 0.4,
        "alpha": 1.0,
        "anisotropy": 0.15,
        "texture_path": _pbr_map("metal", "albedo"),
        "normal_map_path": _pbr_map("metal", "normal"),
        "normal_map_strength": 0.85,
        "roughness_map_path": _pbr_map("metal", "roughness"),
        "ao_map_path": _pbr_map("metal", "ao"),
        "metallic_map_path": _pbr_map("metal", "metallic"),
        "uv_scale_meters": 1.5,
        "shader_variant": _ssr(),
    },
    "wood": {
        "color": [0.266, 0.109, 0.060],
        "roughness": 0.7,
        "metallic": 0.0,
        "reflectance": 0.3,
        "alpha": 1.0,
    },
    "plasterboard": {
        "color": [0.051, 0.539, 0.133],
        "roughness": 0.9,
        "metallic": 0.0,
        "reflectance": 0.2,
        "alpha": 1.0,
    },
    "brick": {
        "color": [0.402, 0.112, 0.087],
        "roughness": 0.85,
        "metallic": 0.0,
        "reflectance": 0.25,
        "alpha": 1.0,
        "texture_path": _pbr_map("brick", "albedo"),
        "normal_map_path": _pbr_map("brick", "normal"),
        "normal_map_strength": 1.55,
        "roughness_map_path": _pbr_map("brick", "roughness"),
        "ao_map_path": _pbr_map("brick", "ao"),
        "uv_scale_meters": 2.0,
    },
    "marble": {
        "color": [0.701, 0.644, 0.485],
        "roughness": 0.3,
        "metallic": 0.0,
        "reflectance": 0.6,
        "alpha": 1.0,
        "clearcoat": 0.6,
        "clearcoat_roughness": 0.1,
        "texture_path": _pbr_map("marble", "albedo"),
        "normal_map_path": _pbr_map("marble", "normal"),
        "normal_map_strength": 0.45,
        "roughness_map_path": _pbr_map("marble", "roughness"),
        "ao_map_path": _pbr_map("marble", "ao"),
        "uv_scale_meters": 2.0,
        "shader_variant": _ssr(),
    },
    "chipboard": {
        "color": [0.509, 0.159, 0.323],
        "roughness": 0.8,
        "metallic": 0.0,
        "reflectance": 0.25,
        "alpha": 1.0,
    },
    "floorboard": {
        "color": [0.539, 0.386, 0.025],
        "roughness": 0.6,
        "metallic": 0.0,
        "reflectance": 0.35,
        "alpha": 1.0,
    },
    "ceiling_board": {
        "color": [0.376, 0.539, 0.117],
        "roughness": 0.85,
        "metallic": 0.0,
        "reflectance": 0.2,
        "alpha": 1.0,
    },
    "plywood": {
        "color": [0.136, 0.076, 0.539],
        "roughness": 0.7,
        "metallic": 0.0,
        "reflectance": 0.3,
        "alpha": 1.0,
    },
    "very_dry_ground": {
        "color": [0.539, 0.319, 0.223],
        "roughness": 0.95,
        "metallic": 0.0,
        "reflectance": 0.15,
        "alpha": 1.0,
        "texture_path": _pbr_map("asphalt", "albedo"),
        "normal_map_path": _pbr_map("asphalt", "normal"),
        "normal_map_strength": 1.15,
        "roughness_map_path": _pbr_map("asphalt", "roughness"),
        "ao_map_path": _pbr_map("asphalt", "ao"),
        "uv_scale_meters": 4.0,
    },
    "medium_dry_ground": {
        "color": [0.539, 0.181, 0.076],
        "roughness": 0.9,
        "metallic": 0.0,
        "reflectance": 0.2,
        "alpha": 1.0,
        "texture_path": _pbr_map("asphalt", "albedo"),
        "normal_map_path": _pbr_map("asphalt", "normal"),
        "normal_map_strength": 1.15,
        "roughness_map_path": _pbr_map("asphalt", "roughness"),
        "ao_map_path": _pbr_map("asphalt", "ao"),
        "uv_scale_meters": 4.0,
    },
    "wet_ground": {
        "color": [0.539, 0.027, 0.147],
        "roughness": 0.7,
        "metallic": 0.0,
        "reflectance": 0.3,
        "alpha": 1.0,
        "texture_path": _pbr_map("asphalt", "albedo"),
        "normal_map_path": _pbr_map("asphalt", "normal"),
        "normal_map_strength": 1.15,
        "roughness_map_path": _pbr_map("asphalt", "roughness"),
        "ao_map_path": _pbr_map("asphalt", "ao"),
        "uv_scale_meters": 4.0,
    },
    "ground_asphalt": {
        "color": [0.18, 0.18, 0.18],
        "roughness": 0.9,
        "metallic": 0.0,
        "reflectance": 0.15,
        "alpha": 1.0,
        "texture_path": _pbr_map("asphalt", "albedo"),
        "normal_map_path": _pbr_map("asphalt", "normal"),
        "normal_map_strength": 1.15,
        "roughness_map_path": _pbr_map("asphalt", "roughness"),
        "ao_map_path": _pbr_map("asphalt", "ao"),
        "uv_scale_meters": 4.0,
    },
    "ground_nist_ctl_floor": {
        "color": [0.20, 0.20, 0.19],
        "roughness": 0.72,
        "metallic": 0.02,
        "reflectance": 0.32,
        "alpha": 1.0,
        "clearcoat": 0.12,
        "clearcoat_roughness": 0.35,
        "anisotropy": 0.0,
        "texture_path": _pbr_map("nist_ctl_floor", "albedo"),
        "normal_map_path": _pbr_map("nist_ctl_floor", "normal"),
        "normal_map_strength": 0.8,
        "roughness_map_path": _pbr_map("nist_ctl_floor", "roughness"),
        "ao_map_path": _pbr_map("nist_ctl_floor", "ao"),
        "metallic_map_path": _pbr_map("nist_ctl_floor", "metallic"),
        "uv_scale_meters": 60.0,
    },
    "vegetation": {
        "color": [0.3, 0.5, 0.2],
        "roughness": 0.9,
        "metallic": 0.0,
        "reflectance": 0.2,
        "alpha": 1.0,
        "texture_path": _pbr_map("grass", "albedo"),
        "normal_map_path": _pbr_map("grass", "normal"),
        "normal_map_strength": 0.75,
        "roughness_map_path": _pbr_map("grass", "roughness"),
        "ao_map_path": _pbr_map("grass", "ao"),
        "uv_scale_meters": 5.0,
    },
    "water": {
        "color": [0.2, 0.4, 0.6],
        "roughness": 0.05,
        "metallic": 0.0,
        "reflectance": 0.8,
        "alpha": 0.85,
        "transmission": 0.6,
        "glass_thickness": 2.0,
        "absorption_color": [0.55, 0.78, 0.92],
        "shader_variant": _ssr(),
    },
    "skin": {
        "color": [0.82, 0.62, 0.49],
        "roughness": 0.55,
        "metallic": 0.0,
        "reflectance": 0.35,
        "alpha": 1.0,
    },
    "default": {
        "color": [0.8, 0.6, 0.5],
        "roughness": 0.6,
        "metallic": 0.0,
        "reflectance": 0.4,
        "alpha": 1.0,
    },
}


def material_id_stem(material_id: object | None, *, strip_suffix: bool = False) -> str:
    """Return the normalized material ID stem used for visual preset inference."""
    if not material_id:
        return ""
    name = str(material_id).strip().lower()
    for prefix in _MATERIAL_ID_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if strip_suffix:
        for suffix in _MATERIAL_ID_SUFFIXES:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
    return name


def infer_material_type_from_id(material_id: object | None) -> str:
    """Infer a catalog material type from a scene material ID."""
    name = material_id_stem(material_id, strip_suffix=True)
    return name if name in ITU_TO_PBR else "default"


def material_preset(material_type: object, *, copy: bool = True) -> dict[str, object]:
    """Return the catalog preset for a material type, falling back to default."""
    key = normalize_material_type_name(material_type)
    preset = ITU_TO_PBR.get(key, ITU_TO_PBR["default"])
    return dict(preset) if copy else preset


def is_known_material_type(material_type: object) -> bool:
    """Return whether a material type resolves to a named catalog preset."""
    key = normalize_material_type_name(material_type, default="")
    return bool(key) and key in ITU_TO_PBR


PBR_WAVE_A_FIELDS = (
    "clearcoat",
    "clearcoat_roughness",
    "anisotropy",
    "emissive_intensity",
)
PBR_WAVE_B_FIELDS = ("transmission", "glass_thickness")
PBR_ADVANCED_FIELDS = PBR_WAVE_A_FIELDS + ("emissive_color",) + PBR_WAVE_B_FIELDS


def effective_emissive_color(color: Any, props: dict) -> tuple[float, float, float]:
    """Resolve emissive color, falling back to base color when intensity is active."""
    emissive_color = tuple(props.get("emissive_color", (0.0, 0.0, 0.0)))
    emissive_intensity = float(props.get("emissive_intensity", 0.0))
    if emissive_intensity > 0.0 and not any(c > 0.0 for c in emissive_color):
        try:
            return (float(color[0]), float(color[1]), float(color[2]))
        except (TypeError, ValueError, IndexError):
            return (1.0, 1.0, 1.0)
    return (
        float(emissive_color[0]),
        float(emissive_color[1]),
        float(emissive_color[2]),
    )


def _resolved_pbr_kwargs(
    color: Any,
    props: Mapping[str, Any],
    texture_policy: TexturePolicyResult,
) -> dict[str, Any]:
    """Build renderer kwargs from properties whose texture paths are resolved."""
    renderer_color = texture_policy.renderer_base_color[:3]
    renderer_alpha = texture_policy.renderer_base_color[3]
    emissive_intensity = float(props.get("emissive_intensity", 0.0))
    emissive_color = effective_emissive_color(color, dict(props))
    return {
        "color": list(renderer_color),
        "color_multiplier": tuple(props.get("color_multiplier", (1.0, 1.0, 1.0))),
        "roughness": props.get("roughness", 0.5),
        "metallic": props.get("metallic", 0.0),
        "reflectance": props.get("reflectance", 0.5),
        "alpha": renderer_alpha,
        "clearcoat": float(props.get("clearcoat", 0.0)),
        "clearcoat_roughness": float(props.get("clearcoat_roughness", 0.0)),
        "anisotropy": float(props.get("anisotropy", 0.0)),
        "emissive_color": emissive_color,
        "emissive_intensity": emissive_intensity,
        "transmission": float(props.get("transmission", 0.0)),
        "glass_thickness": float(props.get("glass_thickness", 0.0)),
        "absorption_color": tuple(props.get("absorption_color", (1.0, 1.0, 1.0))),
        **{key: props.get(key) for key in TEXTURE_MAP_KEYS},
        "normal_map_strength": float(props.get("normal_map_strength", 1.0)),
        "uv_scale_meters": float(props.get("uv_scale_meters", 2.0)),
        "uv_repeat_scale": props.get("uv_repeat_scale"),
        "shader_variant": props.get("shader_variant"),
    }


def resolve_pbr_material(
    color: Any,
    props: Mapping[str, Any],
    *,
    context: str | None = None,
    validate_paths: bool | None = None,
) -> ResolvedMaterial:
    """Resolve one immutable material at the application/renderer boundary."""
    source_props = dict(props)
    paths_already_resolved = bool(source_props.pop(TEXTURE_POLICY_RESOLVED_KEY, False))
    should_validate = not paths_already_resolved if validate_paths is None else validate_paths
    filtered_props, texture_policy = apply_texture_policy_to_props(
        source_props,
        color=color,
        alpha=source_props.get("alpha", 1.0),
        context=context or str(source_props.get("material_type") or "material"),
        validate_paths=should_validate,
    )
    warn_for_texture_policy(texture_policy, log=logger)
    renderer_kwargs = _resolved_pbr_kwargs(color, filtered_props, texture_policy)
    return ResolvedMaterial(
        properties=filtered_props,
        texture_policy=texture_policy,
        payload=material_payload_from_mapping(renderer_kwargs),
    )


def material_payload_to_pbr_kwargs(material: MaterialPayload) -> dict[str, Any]:
    """Return renderer kwargs from an already-resolved immutable payload."""
    return {
        "color": list(material.base_color[:3]),
        "color_multiplier": tuple(material.color_multiplier),
        "roughness": material.roughness,
        "metallic": material.metallic,
        "reflectance": material.reflectance,
        "alpha": material.base_color[3],
        "clearcoat": material.clearcoat,
        "clearcoat_roughness": material.clearcoat_roughness,
        "anisotropy": material.anisotropy,
        "emissive_color": tuple(material.emissive_color),
        "emissive_intensity": material.emissive_intensity,
        "transmission": material.transmission,
        "glass_thickness": material.glass_thickness,
        "absorption_color": tuple(material.absorption_color),
        "texture_path": material.texture_path,
        "normal_map_path": material.normal_map_path,
        "roughness_map_path": material.roughness_map_path,
        "ao_map_path": material.ao_map_path,
        "metallic_map_path": material.metallic_map_path,
        "normal_map_strength": material.normal_map_strength,
        "uv_scale_meters": material.uv_scale_meters,
        "uv_repeat_scale": material.uv_repeat_scale,
        "shader_variant": material.shader_variant,
    }


def pbr_props_to_kwargs(color: Any, props: Mapping[str, Any]) -> dict[str, Any]:
    """Build normalized backend material kwargs from authored properties."""
    resolved = resolve_pbr_material(color, props)
    return material_payload_to_pbr_kwargs(resolved.payload)


def effective_texture_path(props: dict, cache_texture_path: Optional[str]) -> Optional[str]:
    """Resolve albedo texture path, preferring preset over texture cache."""
    preset_path = props.get("texture_path")
    if preset_path:
        return str(preset_path)
    return cache_texture_path
