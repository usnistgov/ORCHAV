"""Optional procedural building appearance; never consumes or changes RF inputs.

Producer-authored face counts distinguish buildings from barriers sharing a scene
material mesh. UVs and textures are derived only for render payloads. Original
buffers stay owned by the scene entry and are restored when styling is disabled.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from shared.logging import get_logger

from ..model import RenderObjectState
from ..types.render_payloads import MeshPayload, SurfaceColorSource
from .texture_policy import TEXTURE_MAP_KEYS, textures_globally_enabled

logger = get_logger("orchav.building_style")
STYLE_VERSION = "building-facades-v2"
TILE_METRES = 3.0
_TEXTURE_SIZE = 256
_MAX_GENERATED_TEXTURES = 128
_ORIGINAL_UV = (0.97, 0.97)


def _facade_exclusions(record: dict[str, Any], count: int) -> tuple[tuple[int, int], ...]:
    """Validate identified whole-part exclusions before enabling any facade.

    Source IDs document the producer's semantic association; the renderer never
    guesses landmarks from filenames or fetches live OSM data. Ranges must match
    the existing complete part partition, including separate roof chunks.
    """
    from .scene_variation import valid_parts

    records = record.get("facade_exclusions", [])
    if not isinstance(records, list):
        raise ValueError("facade_exclusions must be a list")
    if not records:
        return ()
    parts = record.get("parts")
    if not valid_parts(parts, count):
        raise ValueError("facade exclusions require a complete building-part partition")
    available = {tuple(part) for part in parts}
    excluded = set()
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("facade exclusions must identify whole building parts")
        first, size = item.get("first_triangle"), item.get("triangle_count")
        source_id = item.get("source_id")
        if (
            type(first) is not int
            or type(size) is not int
            or (first, size) not in available
            or not isinstance(source_id, str)
            or not source_id.strip()
        ):
            raise ValueError("invalid facade exclusion range or source identity")
        excluded.add((first, size))
    return tuple(sorted(excluded))


def annotate_building_faces(entries: list[dict[str, Any]], xml_path: str) -> None:
    """Attach optional producer classification after both cold and cached scene loads."""
    path = Path(xml_path).with_name("building_visuals.json")
    if not path.is_file():
        path = Path(xml_path).with_name("osm_metadata.json")
    if not path.is_file():
        return
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        records = metadata.get("visual_building_faces", {})
        if not isinstance(records, dict):
            records = {}
        roads = metadata.get("visual_road_faces", {})
        if not isinstance(roads, dict):
            roads = {}
        from .scene_variation import valid_parts, valid_roads

        for entry in entries:
            record = records.get(entry.get("rel_path"), {})
            mesh = entry.get("mesh")
            road_record = roads.get(entry.get("rel_path"))
            if isinstance(mesh, MeshPayload) and valid_roads(road_record, mesh):
                entry["_osm_road_segments"] = road_record["segments"]
            if not isinstance(record, dict) or not isinstance(mesh, MeshPayload):
                continue
            count, total = record.get("building_triangles"), record.get("total_triangles")
            if (
                type(count) is int
                and type(total) is int
                and 0 < count <= total == len(mesh.triangles)
            ):
                try:
                    exclusions = _facade_exclusions(record, count)
                except ValueError as exc:
                    # Fail closed: corrupt protection metadata must never put
                    # ordinary windows back onto a protected landmark.
                    entry.pop("_osm_building_faces", None)
                    entry.pop("_osm_building_parts", None)
                    entry.pop("_osm_facade_exclusions", None)
                    logger.warning(
                        "Keeping original appearance for %s: %s", entry.get("rel_path"), exc
                    )
                    continue
                entry["_osm_building_faces"] = count
                entry["_osm_facade_exclusions"] = exclusions
                if valid_parts(record.get("parts"), count):
                    entry["_osm_building_parts"] = record["parts"]
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        logger.warning("Building appearance metadata unavailable: %s", exc)


def facade_pixels(variant: int, original_rgb: tuple[int, int, int]) -> np.ndarray:
    """Draw an original, tileable 3 m facade; no external artwork is used.

    Small neutral swatches provide roof and unclassified-face colors. They also
    form restrained trim in repeated facades. No transparent or emissive windows.
    """
    from PIL import Image, ImageDraw

    palette = ((218, 208, 189), (165, 106, 83), (173, 180, 182))
    image = Image.new("RGB", (_TEXTURE_SIZE, _TEXTURE_SIZE), palette[variant % 3])
    draw = ImageDraw.Draw(image)
    if variant % 3 == 1:
        for row, y in enumerate(range(0, 256, 16)):
            draw.line((0, y, 255, y), fill=(191, 165, 144), width=2)
            for x in range(-32 if row % 2 else 0, 256, 64):
                draw.line((x, y, x, y + 16), fill=(191, 165, 144), width=2)
    else:
        draw.line((0, 252, 255, 252), fill=(145, 147, 145), width=5)
    for x in (36, 158):
        draw.rectangle((x - 5, 52, x + 66, 187), fill=(117, 120, 116))
        draw.rectangle((x, 57, x + 61, 181), fill=(58, 78, 89))
        draw.line((x + 30, 57, x + 30, 181), fill=(192, 193, 183), width=3)
        draw.line((x, 121, x + 61, 121), fill=(192, 193, 183), width=3)
        draw.rectangle((x - 8, 185, x + 69, 193), fill=(205, 202, 187))
    # PIL rows and UV V coordinates use opposite origins in both renderer paths.
    # Duplicate swatches vertically, making flat face sampling origin-independent.
    for y in (0, 240):
        draw.rectangle((240, y, 255, y + 15), fill=original_rgb)
        draw.rectangle((224, y, 239, y + 15), fill=(103, 108, 112))
    return np.asarray(image)


def _texture(variant: int, color: Any) -> str:
    from PIL import Image

    from ..scene.io import get_uv_cache_root

    rgb = tuple(int(round(float(v) * 255)) for v in np.clip(color[:3], 0, 1))
    root = get_uv_cache_root() / STYLE_VERSION
    root.mkdir(parents=True, exist_ok=True)
    name = f"{variant}-{'-'.join(map(str, rgb))}.png"
    path = root / name
    if not path.is_file():
        fd, temp = tempfile.mkstemp(prefix="facade-", suffix=".png", dir=root)
        os.close(fd)
        try:
            Image.fromarray(facade_pixels(variant, rgb)).save(temp)
            os.replace(temp, path)
            # Bound disk artwork separately from the shared decoded-texture cache.
            files = sorted(root.glob("*.png"), key=lambda item: item.stat().st_mtime_ns)
            for stale in files[:-_MAX_GENERATED_TEXTURES]:
                if stale != path:
                    stale.unlink(missing_ok=True)
        finally:
            Path(temp).unlink(missing_ok=True)
    return str(path)


def styled_payload(source: MeshPayload, count: int, identity: str, *, exclusions=()) -> MeshPayload:
    """Add cached metre-scaled UVs while retaining exact vertices and triangles."""
    from ..scene.io import get_uv_cache_root
    from ..scene.uv_cache_store import get_uv_cache_store

    digest = sha256(f"{STYLE_VERSION}|{count}|{identity}|{TILE_METRES}".encode())
    if exclusions:
        digest.update(json.dumps(exclusions).encode())
    digest.update(np.ascontiguousarray(source.vertices).tobytes())
    digest.update(np.ascontiguousarray(source.triangles).tobytes())
    key = digest.hexdigest()
    store = get_uv_cache_store(get_uv_cache_root(), STYLE_VERSION)
    uvs = store.get(key, source_signature=key, expected_shape=(len(source.triangles) * 3, 2))
    if uvs is None:
        # Facades only need a wall tangent and world-up. Vectorize this instead
        # of finding connected coplanar patches in the general UV projector.
        # Roofs and unclassified faces sample constant swatches below.
        uvs = np.zeros((len(source.triangles), 3, 2), dtype=np.float32)
        corners = source.vertices[source.triangles[:count]]
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        tangent = np.cross(np.array([0.0, 0.0, 1.0]), normals)
        tangent_lengths = np.linalg.norm(tangent, axis=1)
        np.divide(
            tangent, tangent_lengths[:, None], out=tangent, where=tangent_lengths[:, None] > 1e-12
        )
        local = corners - np.min(source.vertices, axis=0)
        uvs[:count, :, 0] = np.einsum("tvc,tc->tv", local, tangent) / TILE_METRES
        uvs[:count, :, 1] = local[:, :, 2] / TILE_METRES
        roofs = (np.abs(normals[:, 2]) > 0.5 * lengths) | (lengths < 1e-12)
        uvs[:count][roofs] = (0.905, 0.03)
        uvs[count:] = _ORIGINAL_UV
        for first, size in exclusions:
            uvs[first : first + size] = _ORIGINAL_UV
        uvs = uvs.reshape(-1, 2)
        store.put(key, uvs, source_signature=key)
        store.flush()
    return replace(source, triangle_uvs=uvs, cache_key=f"{STYLE_VERSION}/{key}")


def apply_building_style(
    entry: dict[str, Any],
    props: dict[str, Any],
    *,
    enabled: bool,
    eligible: bool,
    mode: str = "procedural",
    roads: bool = False,
    theme: str = "mixed",
) -> dict[str, Any]:
    """Resolve opt-in styling below explicit assignments; restore exact originals."""
    mesh = entry.get("mesh")
    if not isinstance(mesh, RenderObjectState) or not isinstance(mesh.payload, MeshPayload):
        return props
    cached = entry.get("_building_style_payloads")
    current = mesh.payload
    if cached is not None:
        original, styled = cached
        # A newly loaded/transformed payload invalidates the per-entry cache.
        if current.triangle_uvs is styled.triangle_uvs:
            if current.vertices is styled.vertices and current.triangles is styled.triangles:
                current = original
            else:
                # Scene edits may preserve UV buffers while changing vertices.
                # Keep the edit, remove our old UVs, and derive a new cache key.
                current = replace(current, triangle_uvs=original.triangle_uvs, cache_key=None)
                cached = None
        elif current is not original:
            cached = None
        if cached is None:
            entry.pop("_building_style_payloads", None)
    requested_mode = "roads" if roads and entry.get("_osm_road_segments") else mode
    if requested_mode == "varied" and not entry.get("_osm_building_parts"):
        requested_mode = "procedural"
    # Facade themes do not invalidate road or simple-facade resources.
    theme = theme if requested_mode == "varied" else "mixed"
    exclusions = () if requested_mode == "roads" else entry.get("_osm_facade_exclusions", ())
    style_mode = (requested_mode, theme, exclusions)
    count = entry.get("_osm_building_faces", 0)
    classified = (
        bool(entry.get("_osm_road_segments"))
        if requested_mode == "roads"
        else (
            type(count) is int
            and 0 < count <= len(current.triangles)
            and (requested_mode != "varied" or bool(entry.get("_osm_building_parts")))
        )
    )
    active = (
        enabled
        and eligible
        and textures_globally_enabled()
        and classified
        and current.color_source is not SurfaceColorSource.VERTEX
        and not any(props.get(k) for k in TEXTURE_MAP_KEYS)
    )
    if not active:
        if mesh.payload is not current:
            mesh.replace_payload(current)
        entry.pop("_building_style_key", None)
        return props
    # An unstyled view can retain its prepared recipe for the return trip.
    # Replace it only when an active style requires different geometry.
    if cached is not None and entry.get("_building_style_mode") != style_mode:
        cached = None
        entry.pop("_building_style_payloads", None)
    identity = str(entry.get("rel_path") or entry.get("name", "building"))
    # Budget failures depend on immutable source geometry and classification,
    # not on presentation refreshes. Do not rebuild the same rejected mesh on
    # every material sync. A source edit or changed recipe permits a new attempt.
    rejection_key = None
    if requested_mode == "varied" and theme == "haussmann":
        from .haussmann_geometry import MAX_RENDER_VERTICES, SubdivisionBudgetExceeded
        from .haussmann_textures import VERSION

        rejection_key = (
            style_mode,
            identity,
            count,
            VERSION,
            MAX_RENDER_VERTICES,
            tuple(tuple(part) for part in entry["_osm_building_parts"]),
        )
        rejected = entry.get("_building_style_rejection")
        if rejected is not None and rejected[0] is current and rejected[1] == rejection_key:
            if mesh.payload is not current:
                mesh.replace_payload(current)
            entry.pop("_building_style_key", None)
            return props
        entry.pop("_building_style_rejection", None)
    try:
        color = props.get("color", (0.7, 0.7, 0.7))
        extra_maps = {}
        if requested_mode == "varied" and theme == "haussmann":
            from .haussmann_geometry import facade_payload
            from .haussmann_textures import texture_paths

            if cached is None:
                styled = facade_payload(
                    current,
                    entry["_osm_building_parts"],
                    identity,
                    exclusions=exclusions,
                )
                cached = (current, styled)
                rgb = tuple(int(round(float(v) * 255)) for v in np.clip(color[:3], 0, 1))
                # Fully classified meshes never sample the original-color swatch.
                # Share their entire atlas set across RF material groups.
                if count == len(current.triangles):
                    rgb = (178, 178, 178)
                entry["_building_style_maps"] = texture_paths(rgb)
            extra_maps = entry["_building_style_maps"]
            path = extra_maps["texture_path"]
        elif requested_mode in {"varied", "roads"}:
            from .scene_variation import varied_payload

            if cached is None:
                metadata = (
                    entry["_osm_road_segments"]
                    if requested_mode == "roads"
                    else entry["_osm_building_parts"]
                )
                styled, path = varied_payload(
                    current,
                    metadata,
                    identity,
                    color,
                    roads=requested_mode == "roads",
                    theme=theme,
                    exclusions=exclusions,
                )
                cached = (current, styled)
                entry["_building_style_path"] = path
            path = entry["_building_style_path"]
        else:
            if cached is None:
                cached = (current, styled_payload(current, count, identity, exclusions=exclusions))
            variant = int.from_bytes(sha256(identity.encode()).digest()[:2], "little") % 3
            path = _texture(variant, color)
        entry["_building_style_payloads"] = cached
        entry["_building_style_mode"] = style_mode
        styled = cached[1]
    except (OSError, ValueError) as exc:
        if rejection_key is not None and isinstance(exc, SubdivisionBudgetExceeded):
            entry["_building_style_rejection"] = (current, rejection_key)
        logger.warning("Keeping original building appearance for %s: %s", identity, exc)
        if mesh.payload is not current:
            mesh.replace_payload(current)
        entry.pop("_building_style_key", None)
        return props
    if mesh.payload.triangle_uvs is not styled.triangle_uvs:
        mesh.replace_payload(styled)
    entry["_building_style_key"] = styled.cache_key
    return {
        **props,
        "texture_path": path,
        "color": [1.0, 1.0, 1.0],
        "uv_scale_meters": TILE_METRES,
        "roughness": 1.0 if extra_maps else 0.88,
        **extra_maps,
        "metallic": 0.0,
        "reflectance": 0.3,
        "clearcoat": 0.0,
        "transmission": 0.0,
    }
