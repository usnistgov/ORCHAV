"""Built-in scene recipes shared by desktop and notebook views."""

from ..lighting.assets import cached_environment, environment_catalog
from ..renderers.effect_settings import DEFAULT_BLOOM
from ..scene.defaults import SCENE_BACKGROUND_PRESETS
from .view_styles import PRESETS


def appearance_recipe(
    name: str, current: dict, *, varied_buildings: bool = False
) -> tuple[dict, str]:
    """Resolve a built-in look for either viewport without mutating its owner."""
    data = dict(current)
    settings = PRESETS[name]
    data.update(settings)
    data["plan_view"] = name == "Plan"
    data["background"] = SCENE_BACKGROUND_PRESETS[settings["background"]]
    data.update(
        alpha=1.0,
        skybox=False,
        color_fidelity=False,
        aa="fxaa",
        mipmaps=True,
        edge_width=1.0,
        bloom={
            **DEFAULT_BLOOM,
            "enabled": name == "Glow",
            "bloom_strength": 0.18,
            "max_mip_levels": 5,
            "filter_radius": 0.003,
        },
    )
    # Keep the scene's regional theme even while maps are hidden.
    if data["textures"]:
        data["building_appearance"] = "varied" if varied_buildings else "procedural"
        data["road_appearance"] = "marked"
        data["authored_surface_textures"] = True
    data["lights"] = dict(
        lighting_profile="custom",
        ambient_intensity=0.8,
        headlight_enabled=False,
        headlight_intensity=1.2,
        key_azimuth_deg=31.0,
        key_elevation_deg=-42.0,
        key_intensity=2.0,
        fill_azimuth_deg=-149.0,
        fill_elevation_deg=-60.0,
        fill_intensity=0.5,
        shadows_enabled=name == "Detailed",
    )
    data["ibl"] = 0
    if name in {"Dark", "Glow"}:
        data["lights"].update(
            ambient_intensity=0.65,
            key_intensity=2.2,
            key_azimuth_deg=38.66,
            key_elevation_deg=-57.37,
            fill_azimuth_deg=-153.43,
            fill_elevation_deg=-33.85,
            shadows_enabled=True,
        )
    notice = ""
    if name == "Detailed":
        catalog = environment_catalog()
        sky = "sky_sunny"
        if cached_environment(catalog[sky]) is not None:
            data["sky"], data["skybox"] = sky, True
        else:
            notice = "Sunny sky not installed. Download Sunny under Lighting and environment, then select Detailed again. Surface detail remains available."
    return data, notice
