"""Portable map reference, projected into the generated scene's UTM frame.

This is Builder display metadata, never ray-tracing geometry. The nine tiles
already requested by the user are reused; saving triggers no tile downloads.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..scene.map_reference import read_map
from ..types.render_payloads import MaterialPayload
from .assets import FileRevision, ScenePreviewAsset
from .domain import SceneReference
from .osm_region import GroundMap

MAP_NAME = "OpenStreetMap reference"
MAP_DIRECTORY = "builder_map"


def save_map_reference(scene_root: Path, reference: dict) -> None:
    """Keep the displayed atlas and georeferencing next to a finished OSM scene."""
    texture = Path(reference["texture"])
    metadata = json.loads((scene_root / "osm_metadata.json").read_text(encoding="utf-8"))
    ground = GroundMap(**reference["map"])
    from ..scene.map_reference import project_map

    ox, oy = ground.origin
    west, south, east, north = ground.bounds
    atlas_bounds = (
        west / ground.scale + ox,
        south / ground.scale + oy,
        east / ground.scale + ox,
        north / ground.scale + oy,
    )
    south_lat, west_lon, north_lat, east_lon = metadata["bbox"]
    west, south = ground.world(south_lat, west_lon)
    east, north = ground.world(north_lat, east_lon)
    pad = float(metadata.get("osm_config", {}).get("ground_pad_m", 0))
    bounds = (
        (west - pad) / ground.scale + ox,
        (south - pad) / ground.scale + oy,
        (east + pad) / ground.scale + ox,
        (north + pad) / ground.scale + oy,
    )
    geometry = project_map(metadata, atlas_bounds, bounds=bounds, height=0.08)
    destination = scene_root / MAP_DIRECTORY
    destination.mkdir(exist_ok=True)
    shutil.copy2(texture, destination / "map.png")
    (destination / "map.json").write_text(
        json.dumps(
            {
                **geometry,
                "attribution": reference.get("attribution", "© OpenStreetMap contributors"),
            }
        ),
        encoding="utf-8",
    )


def load_map_reference(path: Path, scene: SceneReference) -> ScenePreviewAsset:
    """Load a saved display reference without contacting a map provider."""
    mesh, texture_name, attribution = read_map(path)
    texture = Path(texture_name)
    key = f"builder-map:{FileRevision.capture(path).token}:{FileRevision.capture(texture).token}"
    return ScenePreviewAsset(
        cache_key=key,
        name=MAP_NAME,
        scene_source=scene.source,
        scene_id=scene.id,
        scene_path=str(path),
        mesh_path=None,
        attribution=attribution,
        mesh=mesh,
        material=MaterialPayload(
            shader="unlit",
            base_color=(1, 1, 1, 1),
            texture_path=str(texture),
            shader_variant="reference_image",
        ),
    )
