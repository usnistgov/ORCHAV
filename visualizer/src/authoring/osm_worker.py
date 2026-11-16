"""Isolated map/scene jobs. stdout is JSONL; third-party output stays in the job log."""

from __future__ import annotations

import contextlib
import json
import logging
import sys
import time
from pathlib import Path

from ..scene.map_tiles import TILE_URL, fetch_tile, tile_expiry  # noqa: F401
from .osm_region import GroundMap, validate_bbox


def emit(event, **values):
    sys.stdout.write(json.dumps({"event": event, **values}) + "\n")
    sys.stdout.flush()


def load_map(config, root, cancelled, progress):
    import requests
    from PIL import Image, ImageDraw

    ground = GroundMap(**config["map"])
    tx, ty = ground.tile_origin
    atlas = Image.new("RGB", (768, 768), "#e2e7e9")
    cache = Path(config["cache"])
    template = config.get("tile_url", TILE_URL)
    loaded = cached = 0
    errors = []
    with requests.Session() as session:
        for row in range(3):
            for col in range(3):
                if cancelled():
                    raise InterruptedError("Map loading cancelled")
                url = template.format(z=ground.zoom, x=tx + col, y=ty + row)
                try:
                    tile, reused = fetch_tile(session, url, cache, cancelled)
                    atlas.paste(tile, (col * 256, row * 256))
                    loaded += 1
                    cached += int(reused)
                except InterruptedError:
                    raise
                except Exception as exc:
                    errors.append(str(exc))
                    ImageDraw.Draw(atlas).text(
                        (col * 256 + 15, row * 256 + 115), "Tile unavailable", fill="#576875"
                    )
                progress({"stage": "map", "message": f"Map tiles: {row * 3 + col + 1}/9"})
    output = root / "map.png"
    atlas.save(output)
    return {
        "texture": str(output),
        "loaded_tiles": loaded,
        "cached_tiles": cached,
        "errors": errors,
    }


def build_scene(config, root, cancelled, progress):
    from importlib.util import find_spec

    missing = [
        name
        for name in ("osmnx", "pyproj", "shapely", "triangle", "geopandas")
        if find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(
            "OSM building needs optional dependencies: "
            + ", ".join(missing)
            + ". Install ORCHAV's osm extra in the running Python environment, or set ORCHAV_OSM_PYTHON to an interpreter that has it."
        )
    from generator.osm.metadata import validate_generated_scene
    from generator.osm.scenario_extension import build_configured_scene

    config = dict(config)
    reference = config.pop("_ground_map", None)
    validate_bbox(config["bbox"])
    scene_root = root / "scene"
    progress({"message": "Preparing OSM downloads and cached layers…"})
    build_configured_scene(config, scene_root, cancel=cancelled, progress=progress)
    complete, reason = validate_generated_scene(scene_root, config)
    if not complete:
        raise RuntimeError(
            f"Scene acquisition was incomplete: {reason}. Retry to reuse successful downloads."
        )
    warnings = []
    if reference:
        from .map_overlay import save_map_reference

        try:
            save_map_reference(scene_root, reference)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warnings.append(f"Scene built, but the map reference could not be saved: {exc}")
    metadata = json.loads((scene_root / "osm_metadata.json").read_text(encoding="utf-8"))
    acquisition = metadata.get("acquisition", {})
    return {
        "scene_root": str(scene_root),
        "statistics": {
            "meshes": len(metadata.get("artifacts", {}).get("mesh_files", [])),
            "cache_hits": metadata.get("osm_cache_stats", {}).get("hits", 0),
            "requests": len(acquisition.get("requests", [])),
            "layers": len(acquisition.get("layers", {})),
        },
        "warnings": warnings,
    }


def main():
    request_path = Path(sys.argv[1]).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    root = request_path.parent
    started = time.monotonic()

    def cancelled():
        return (root / "cancel").exists()

    # Keep stdout reserved even when third-party generators print directly.
    def progress(value):
        with contextlib.redirect_stdout(sys.__stdout__):
            emit("progress", detail=value)

    class ProgressLog(logging.Handler):
        def emit(self, record):
            if record.name.startswith("generator.osm"):
                progress({"stage": "build", "message": record.getMessage()})

    logger = logging.getLogger("generator.osm")
    logger.setLevel(logging.INFO)
    logger.addHandler(ProgressLog())

    try:
        with contextlib.redirect_stdout(sys.stderr):
            from ..scene.map_reference import load_scene_map

            operation = {"map": load_map, "scene_map": load_scene_map, "build": build_scene}[
                request["operation"]
            ]
            result = operation(request["config"], root, cancelled, progress)
        if cancelled():
            raise InterruptedError("Operation cancelled")
        emit("complete", result=result, elapsed_s=time.monotonic() - started)
    except Exception as exc:
        import traceback

        traceback.print_exc(file=sys.stderr)
        emit("failed", error=str(exc), elapsed_s=time.monotonic() - started)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
