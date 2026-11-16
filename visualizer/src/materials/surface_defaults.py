"""Display defaults for producer-classified scene surfaces, independent of RF inputs."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from shared.scenarios.paths import find_asset_project_root

from .catalog import material_preset


def geographic_surface_properties(
    mesh_path: str | None, radio_material: str
) -> dict[str, Any] | None:
    """Return surface appearance for the producer's known mesh roles.

    Call only for scenes accompanied by producer surface metadata. A solid building
    volume has no modeled interior, so its glass facade stays opaque; thin
    windows in ordinary imported scenes retain their authored transparency.
    """
    name = PurePosixPath(str(mesh_path or "").replace("\\", "/")).name.lower()
    if name == "ground.ply":
        props = dict(material_preset("concrete"))
        props.update(color=[0.43, 0.44, 0.42], roughness=0.94, reflectance=0.22)
    elif name == "streets.ply":
        props = dict(material_preset("ground_asphalt"))
        props.update(color=[0.075, 0.085, 0.095], roughness=0.93, reflectance=0.18)
    elif name == "buildings_glass.ply" and radio_material == "glass":
        props = dict(material_preset("glass"))
        props.update(
            color=[0.32, 0.40, 0.45],
            alpha=1.0,
            transmission=0.0,
            roughness=0.22,
            reflectance=0.65,
            clearcoat=0.25,
            clearcoat_roughness=0.18,
        )
    elif name == "water.ply":
        maps = (
            find_asset_project_root(Path(__file__).parent) / "libraries/textures/pbr/water_ripples"
        )
        props = dict(material_preset("glass"))
        props.update(
            color=[36 / 255, 63 / 255, 68 / 255],
            alpha=1.0,
            transmission=0.0,
            roughness=0.45,
            # The renderer uses F0=0.16*r²; this yields water's IOR of 1.333.
            reflectance=(1.333 - 1) / (1.333 + 1) / 0.4,
            texture_path=str(maps / "albedo.png") if (maps / "albedo.png").is_file() else None,
            normal_map_path=str(maps / "normal.png") if (maps / "normal.png").is_file() else None,
            roughness_map_path=(
                str(maps / "roughness.png") if (maps / "roughness.png").is_file() else None
            ),
            normal_map_strength=0.8,
            uv_scale_meters=256.0,
        )
    else:
        return None
    return props
