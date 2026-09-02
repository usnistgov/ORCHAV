"""Catalog-backed PBR property sync for visualizer scene entries."""

from __future__ import annotations

from typing import Any

from ..materials.catalog import material_preset


def sync_entry_pbr_properties_from_catalog(entry: dict[str, Any], material_type: str) -> None:
    """Update renderer-neutral PBR properties on *entry* for *material_type*."""
    pbr = material_preset(material_type)
    entry["pbr_color"] = list(pbr.get("color", [0.7, 0.7, 0.7]))
    entry["pbr_roughness"] = pbr.get("roughness", 0.5)
    entry["pbr_metallic"] = pbr.get("metallic", 0.0)
    entry["pbr_reflectance"] = pbr.get("reflectance", 0.5)
    entry["pbr_properties"] = {
        "color": list(pbr.get("color", [0.7, 0.7, 0.7])),
        "roughness": pbr.get("roughness", 0.5),
        "metallic": pbr.get("metallic", 0.0),
        "reflectance": pbr.get("reflectance", 0.5),
        "alpha": pbr.get("alpha", 1.0),
        "clearcoat": pbr.get("clearcoat", 0.0),
        "clearcoat_roughness": pbr.get("clearcoat_roughness", 0.0),
        "anisotropy": pbr.get("anisotropy", 0.0),
        "emissive_color": tuple(pbr.get("emissive_color", (0.0, 0.0, 0.0))),
        "emissive_intensity": pbr.get("emissive_intensity", 0.0),
        "transmission": pbr.get("transmission", 0.0),
        "glass_thickness": pbr.get("glass_thickness", 0.0),
        "absorption_color": tuple(pbr.get("absorption_color", (1.0, 1.0, 1.0))),
        "texture_path": pbr.get("texture_path"),
        "normal_map_path": pbr.get("normal_map_path"),
        "normal_map_strength": pbr.get("normal_map_strength", 1.0),
        "roughness_map_path": pbr.get("roughness_map_path"),
        "ao_map_path": pbr.get("ao_map_path"),
        "metallic_map_path": pbr.get("metallic_map_path"),
        "uv_scale_meters": pbr.get("uv_scale_meters", 2.0),
        "uv_repeat_scale": pbr.get("uv_repeat_scale"),
        "shader_variant": pbr.get("shader_variant"),
    }
