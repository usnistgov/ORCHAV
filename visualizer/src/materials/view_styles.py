"""Display-only surface styles; authored materials and radio properties stay intact."""

from dataclasses import replace
from functools import lru_cache
from weakref import ref

import numpy as np

from ..types.render_payloads import MaterialPayload, MeshPayload, SurfaceColorSource
from .catalog import material_preset


@lru_cache(maxsize=1024)
def styled_surface(
    material: MaterialPayload,
    color_source: SurfaceColorSource,
    style: str,
    unlit: bool,
    material_type: str,
    target: bool,
    ground: bool = False,
) -> tuple[MaterialPayload, SurfaceColorSource]:
    """Resolve a cached display layer without modifying the canonical material.

    Neutral scene styles keep actor colors. Radio-material colors apply only
    to the static environment; targets retain their authored visual identity.
    This layer precedes selection highlighting and never processes RF overlays.
    """
    if style == "original" and not unlit:
        return material, color_source
    if style != "original" and not target:
        color = (0.66, 0.69, 0.72)
        if style == "radio":
            color = tuple(material_preset(material_type, copy=False)["color"])
        elif style in {"dark", "glow"}:
            if ground:
                color = (0.12, 0.17, 0.23)
            elif material_type in {"metal", "itu_metal"}:
                color = (0.23, 0.29, 0.37)
            else:
                color = (0.39, 0.46, 0.54)
            if style == "glow":
                color = tuple(channel * tint for channel, tint in zip(color, (0.75, 0.60, 0.85)))
        material = MaterialPayload(
            base_color=(*color, material.base_color[3]),
            shader="unlit" if unlit else "lit",
            roughness=0.95 if style in {"dark", "glow"} else 0.9,
            reflectance=0.15 if style in {"dark", "glow"} else 0.5,
        )
        color_source = SurfaceColorSource.MATERIAL
    elif unlit:
        material = replace(material, shader="unlit")
    return material, color_source


def dark_presentation_ground(entry: dict, payload: MeshPayload) -> bool:
    """Shade roads and thin concrete/asphalt meshes as muted context.

    Untagged concrete/asphalt meshes use a display-only height heuristic.
    Cache only the display classification; immutable geometry and RF materials
    remain unchanged. A replaced vertex buffer invalidates the cached height.
    """
    if entry.get("_osm_road_segments"):
        return True
    if entry.get("material_type") not in {"concrete", "itu_concrete", "asphalt", "itu_asphalt"}:
        return False
    cached = entry.get("_presentation_height")
    if cached is None or cached[0]() is not payload.vertices:
        height = float(np.ptp(payload.vertices[:, 2])) if len(payload.vertices) else float("inf")
        cached = entry["_presentation_height"] = (ref(payload.vertices), height)
    return cached[1] < 1.0


# Everyday scene, publication, and the two presentation styles, in UI order.
PRESETS = {
    "Detailed": dict(
        surface="original",
        unlit=False,
        wireframe=False,
        textures=True,
        background="Light",
        outlines=False,
    ),
    "Paper": dict(
        surface="radio",
        unlit=True,
        wireframe=False,
        textures=False,
        background="White",
        outlines=True,
    ),
    "Plan": dict(
        surface="original",
        unlit=True,
        wireframe=False,
        textures=False,
        background="Plan",
        outlines=False,
    ),
    "Dark": dict(
        surface="dark",
        unlit=False,
        wireframe=False,
        textures=False,
        background="Navy",
        outlines=False,
    ),
    "Glow": dict(
        surface="glow",
        unlit=False,
        wireframe=False,
        textures=False,
        background="Violet",
        outlines=False,
    ),
}

PRESET_ALIASES = {
    "Detailed inspection": "Detailed",
    "Outdoor presentation": "Detailed",
    "Dark presentation": "Dark",
}

DESCRIPTIONS = {
    "Plan": "Builder-style street lines on a metric grid, with trajectories enabled. Keeps your camera view. Uses the saved street graph without downloading a map; leaving Plan restores your previous trajectory visibility.",
    "Detailed": "Authored textures and available procedural facades, with a sunny sky and matching lighting.",
    "Paper": "Radio-material colors, fine edges and a white background, without shadows or bloom.",
    "Dark": "Matte blue-gray surroundings and soft shadows.",
    "Glow": "Violet surroundings and bloom.",
}
