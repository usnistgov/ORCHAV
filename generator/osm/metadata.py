"""Lightweight metadata helpers for generated OSM scenarios.

This module intentionally avoids importing heavy optional OSM generation
dependencies so loader code and tests can reason about generated scenes without
requiring ``osmnx`` or ``open3d``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

OSM_METADATA_FILENAME = "osm_metadata.json"
OSM_METADATA_SCHEMA_VERSION = 5


def normalize_osm_config(osm_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize an OSM YAML config block for stable metadata comparison."""
    cfg = dict(osm_cfg or {})

    center = cfg.get("center")
    bbox = cfg.get("bbox")
    width_m = cfg.get("width_m")

    return {
        "center": [float(v) for v in center] if center is not None else None,
        "bbox": [float(v) for v in bbox] if bbox is not None else None,
        "size_m": float(cfg.get("size_m", 500.0)),
        "width_m": float(width_m) if width_m is not None else None,
        "terrain": bool(cfg.get("terrain", False)),
        "terrain_zoom": int(cfg.get("terrain_zoom", 14)),
        "ground_pad_m": float(cfg.get("ground_pad_m", 200.0)),
        "ground_resolution_m": float(cfg.get("ground_resolution_m", 50.0)),
        "osm_server": cfg.get("osm_server"),
        "barriers": bool(cfg.get("barriers", True)),
        "vegetation": bool(cfg.get("vegetation", True)),
        "bridges": bool(cfg.get("bridges", True)),
        "height_enrichment": bool(cfg.get("height_enrichment", True)),
        # External vegetation enrichment stays opt-in; green styling defaults on.
        "canopy_heights": bool(cfg.get("canopy_heights", False)),
        "tree_rows": bool(cfg.get("tree_rows", False)),
        "vegetation_gap_fill": bool(cfg.get("vegetation_gap_fill", False)),
        "worldcover_water_check": bool(cfg.get("worldcover_water_check", False)),
        "green_vegetation": bool(cfg.get("green_vegetation", True)),
        # Numeric vegetation knob: scenarios that share every other key but
        # differ on tree_row_spacing_m must rebuild the geometry, so the
        # value contributes to the cache key like the bool flags above.
        "tree_row_spacing_m": float(cfg.get("tree_row_spacing_m", 8.0)),
        # Landuse-informed material defaults. Default is True; contributes
        # to the cache key so toggling it forces a geometry rebuild.
        "landuse_materials": bool(cfg.get("landuse_materials", True)),
    }


def load_osm_metadata(path: str | Path) -> dict[str, Any] | None:
    """Load generated OSM metadata from disk."""
    meta_path = Path(path)
    if not meta_path.exists():
        return None

    try:
        with meta_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def should_reuse_generated_scene(
    scenario_root: str | Path,
    osm_cfg: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    """Return whether an existing generated OSM scene can be safely reused."""
    if (osm_cfg or {}).get("refresh_cache") or os.environ.get(
        "OSM_REFRESH_CACHE", ""
    ).strip().lower() in {"1", "true", "yes"}:
        return False, "explicit OSM cache refresh requested"
    return validate_generated_scene(scenario_root, osm_cfg)


def validate_generated_scene(
    scenario_root: str | Path,
    osm_cfg: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    """Check completed artifacts independently of the request to refresh their cache.

    A refresh prevents reusing a previous build, but must not invalidate the
    completed replacement when the Builder adopts or previews it.
    """
    root = Path(scenario_root)
    scene_xml = root / "scene.xml"
    if not scene_xml.exists():
        return False, "scene.xml missing"

    meta = load_osm_metadata(root / OSM_METADATA_FILENAME)
    if meta is None:
        return False, "osm_metadata.json missing"

    if meta.get("schema_version") != OSM_METADATA_SCHEMA_VERSION:
        found = meta.get("schema_version")
        return False, f"metadata schema mismatch ({found!r})"

    if meta.get("osm_config") != normalize_osm_config(osm_cfg):
        return False, "OSM config changed"

    if not (meta.get("acquisition") or {}).get("complete"):
        return False, "previous OSM acquisition incomplete; retry missing layers"

    attempt = load_osm_metadata(root / "osm_acquisition_attempt.json")
    if (
        attempt
        and attempt.get("outcome") != "complete"
        and attempt.get("osm_config") in (None, normalize_osm_config(osm_cfg))
    ):
        return False, "previous rebuild did not complete; retry acquisition"

    artifacts = meta.get("artifacts") or {}
    required_files = [artifacts.get("scene_xml", "scene.xml")]
    street_graph = artifacts.get("street_network_graphml", "street_network.graphml")
    if street_graph:
        required_files.append(street_graph)
    required_files.extend(artifacts.get("mesh_files", []))

    for rel_path in required_files:
        if rel_path and not (root / rel_path).exists():
            return False, f"missing generated artifact: {rel_path}"

    return True, "metadata matched"
