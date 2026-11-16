"""Scenario-loader extension for explicit OSM scene recipes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared.scenarios.extensions import SceneSourceExtension, register_scene_source_extension

from .metadata import normalize_osm_config, should_reuse_generated_scene

logger = logging.getLogger(__name__)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OsmSceneConfigModel(_StrictModel):
    """Parameters for OSM scene generation."""

    center: list[float] | None = None
    bbox: list[float] | None = None
    size_m: float = 500.0
    width_m: float | None = None
    terrain: bool = False
    terrain_zoom: int = 14
    ground_pad_m: float = 200.0
    ground_resolution_m: float = 50.0
    osm_server: str | None = None
    barriers: bool = True
    vegetation: bool = True
    bridges: bool = True
    height_enrichment: bool = True
    url: str | None = None
    overpass_timeout: int | None = Field(default=None, gt=0)
    acquisition_timeout: float = Field(default=180.0, gt=0, allow_inf_nan=False)
    refresh_cache: bool = False
    canopy_heights: bool = False
    tree_rows: bool = False
    vegetation_gap_fill: bool = False
    worldcover_water_check: bool = False
    green_vegetation: bool = True
    tree_row_spacing_m: float = Field(default=8.0, gt=0.0)
    landuse_materials: bool = True

    @model_validator(mode="after")
    def validate_location(self):
        if self.center is None and self.bbox is None:
            raise ValueError("Either 'center' or 'bbox' must be specified for OSM scenes")
        if self.center is not None and len(self.center) != 2:
            raise ValueError("'center' must be [lat, lon] (2 elements)")
        if self.bbox is not None and len(self.bbox) != 4:
            raise ValueError("'bbox' must be [min_lat, min_lon, max_lat, max_lon] (4 elements)")
        return self


def _validate_scene(scene_spec: dict[str, Any]) -> None:
    osm_cfg = scene_spec.get("osm")
    if osm_cfg is None:
        raise ValueError("'osm' config block is required when scene source is 'osm'")
    if not isinstance(osm_cfg, dict):
        raise ValueError("'osm' config block must be a mapping")
    OsmSceneConfigModel(**osm_cfg)


def build_configured_scene(
    osm_cfg: dict[str, Any], scenario_root: Path, *, cancel=None, progress=None
) -> Path:
    """Build explicitly, with the same options as the loader and optional job callbacks."""
    OsmSceneConfigModel(**osm_cfg)
    normalized_osm_cfg = normalize_osm_config(osm_cfg)
    try:
        from generator.osm import build_scene, build_scene_from_center
    except ImportError as exc:
        raise ImportError(
            "OSM scene generation requires osmnx, pyproj, shapely, "
            "triangle, and open3d. Install the optional OSM dependencies "
            "before using scene.source: osm."
        ) from exc

    common_kwargs = {
        "cancel": cancel,
        "progress": progress,
        "osm_acquisition_timeout": osm_cfg.get("acquisition_timeout", 180.0),
        "terrain": osm_cfg.get("terrain", False),
        "terrain_zoom": osm_cfg.get("terrain_zoom", 14),
        "ground_pad_m": osm_cfg.get("ground_pad_m", 200.0),
        "ground_resolution_m": osm_cfg.get("ground_resolution_m", 50.0),
        "osm_server": osm_cfg.get("osm_server"),
        "barriers": osm_cfg.get("barriers", True),
        "vegetation": osm_cfg.get("vegetation", True),
        "bridges": osm_cfg.get("bridges", True),
        "height_enrichment": osm_cfg.get("height_enrichment", True),
        "canopy_heights": bool(osm_cfg.get("canopy_heights", False)),
        "tree_rows": bool(osm_cfg.get("tree_rows", False)),
        "vegetation_gap_fill": bool(osm_cfg.get("vegetation_gap_fill", False)),
        "worldcover_water_check": bool(osm_cfg.get("worldcover_water_check", False)),
        "green_vegetation": bool(osm_cfg.get("green_vegetation", True)),
        "landuse_materials": bool(osm_cfg.get("landuse_materials", True)),
    }
    if "tree_row_spacing_m" in osm_cfg:
        common_kwargs["tree_row_spacing_m"] = float(osm_cfg["tree_row_spacing_m"])
    if osm_cfg.get("overpass_timeout") is not None:
        common_kwargs["osm_overpass_timeout"] = int(osm_cfg["overpass_timeout"])
    if osm_cfg.get("refresh_cache"):
        common_kwargs["osm_refresh_cache"] = bool(osm_cfg["refresh_cache"])

    center = osm_cfg.get("center")
    bbox = osm_cfg.get("bbox")
    if center:
        scene_xml = Path(
            build_scene_from_center(
                lat=center[0],
                lon=center[1],
                size_m=osm_cfg.get("size_m", 500.0),
                output_dir=str(scenario_root),
                width_m=osm_cfg.get("width_m"),
                metadata_config=normalized_osm_cfg,
                **common_kwargs,
            )
        )
    elif bbox:
        scene_xml = Path(
            build_scene(
                bbox=tuple(bbox),
                output_dir=str(scenario_root),
                metadata_config=normalized_osm_cfg,
                **common_kwargs,
            )
        )
    else:
        raise ValueError("OSM scene config must specify 'center' or 'bbox'")
    return scene_xml


def _parse_scene(
    scene_spec: dict[str, Any],
    scenario_root: Path,
    project_root: Path,
) -> tuple[str, str, Path | None]:
    del project_root

    osm_cfg = scene_spec.get("osm", {})
    scene_xml_candidate = scenario_root / "scene.xml"
    can_reuse, reuse_reason = should_reuse_generated_scene(scenario_root, osm_cfg)

    if can_reuse:
        logger.info("OSM scene already exists, reusing: %s", scene_xml_candidate)
        scene_xml = scene_xml_candidate
    else:
        if scene_xml_candidate.exists():
            logger.info("Rebuilding generated OSM scene (%s)", reuse_reason)
        scene_xml = build_configured_scene(osm_cfg, scenario_root)
        logger.info("OSM scene generated: %s", scene_xml)

    configured_scene_id = str(scene_spec.get("id", "") or "").strip()
    scene_id = (
        configured_scene_id
        if configured_scene_id and configured_scene_id != "default"
        else "scene.xml"
    )
    return scene_id, "osm", scene_xml


def register() -> None:
    register_scene_source_extension(
        SceneSourceExtension(source="osm", validate=_validate_scene, parse=_parse_scene)
    )
