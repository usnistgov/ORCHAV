"""Bounded facade and road atlases derived from explicit producer classification.

All coordinates belong to render payloads. Source vertices, topology and radio
materials remain untouched. Atlas bins trade exact architectural dimensions for
bounded memory; these are illustrative styles, never surveyed facade/lane data.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np

from ..types.render_payloads import MeshPayload

VERSION = "scene-variation-v3"
THEMES = ("mixed", "american", "european", "asian", "portuguese", "paris", "haussmann")
FLOORS = (1, 2, 3, 4, 6, 8, 12, 18)
BAYS = (1, 2, 4, 8, 16)
LENGTHS = (8, 16, 32, 64, 128, 256)
# Only explicit surface tags select pedestrian/unpaved styling. Missing or
# mixed tags retain original appearance; architecture themes never classify land.
SURFACES = {
    "concrete": 0,
    "concrete:plates": 0,
    "concrete:lanes": 0,
    "paving_stones": 1,
    "sett": 2,
    "cobblestone": 2,
    "unhewn_cobblestone": 2,
    "gravel": 3,
    "fine_gravel": 3,
    "compacted": 3,
    "dirt": 4,
    "earth": 4,
    "ground": 4,
    "mud": 4,
    "asphalt": 5,
    "sand": 6,
}
_UNPAVED = frozenset({3, 4, 6})
_MOTOR_ROADS = frozenset(
    {
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "residential",
        "unclassified",
        "service",
        "living_street",
        "motorway_link",
        "trunk_link",
        "primary_link",
        "secondary_link",
        "tertiary_link",
    }
)


def valid_parts(parts: object, count: int) -> bool:
    """Require a complete, ordered partition of the classified building faces."""
    if not isinstance(parts, list) or not parts:
        return False
    end = 0
    for item in parts:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(type(v) is not int for v in item)
            or item[0] != end
            or item[1] <= 0
        ):
            return False
        end += item[1]
    return end == count


def valid_roads(record: object, mesh: MeshPayload) -> bool:
    """Check ordered producer quads and finite arc lengths before mapping them."""
    if not isinstance(record, dict) or record.get("total_triangles") != len(mesh.triangles):
        return False
    segments = record.get("segments")
    if not isinstance(segments, list) or not segments or len(segments) * 2 != len(mesh.triangles):
        return False
    for i, segment in enumerate(segments):
        if not isinstance(segment, dict) or segment.get("first_triangle") != 2 * i:
            return False
        if (
            type(segment.get("first_triangle")) is not int
            or not isinstance(segment.get("highway"), str)
            or not isinstance(segment.get("lanes"), str)
            or type(segment.get("oneway")) is not bool
            or any(
                type(segment.get(k)) not in (int, float)
                for k in ("start_m", "end_m", "length_m", "width_m")
            )
        ):
            return False
        try:
            a, b, length, width = [
                float(segment[k]) for k in ("start_m", "end_m", "length_m", "width_m")
            ]
        except (KeyError, TypeError, ValueError):
            return False
        if not np.isfinite([a, b, length, width]).all() or not (
            0 <= a < b <= length + 1e-5 and width > 0
        ):
            return False
        triangles = mesh.triangles[2 * i : 2 * i + 2]
        if (
            triangles[0, 0] != triangles[1, 0]
            or triangles[0, 2] != triangles[1, 1]
            or len(np.unique(triangles)) != 4
        ):
            return False
    return True


def _nearest(values: tuple, value: float) -> int:
    return int(np.argmin(np.abs(np.asarray(values) - value)))


def _facade_tile(style: int, floors: int, bays: int, theme: str):
    from PIL import Image, ImageDraw

    if theme == "portuguese":
        return _portuguese_facade_tile(style, floors, bays)
    if theme == "paris":
        from .paris_facades import facade_tile

        return facade_tile(style, floors, bays)

    colors = (
        (213, 199, 173),
        (150, 86, 64),
        (190, 182, 162),
        (158, 166, 172),
        (73, 103, 117),
        (119, 126, 125),
    )
    if theme == "european":
        colors = (
            (224, 207, 175),
            (167, 103, 80),
            (211, 201, 183),
            (183, 174, 153),
            (111, 128, 132),
            (143, 143, 135),
        )
    elif theme == "american":
        colors = (
            (192, 180, 155),
            (132, 73, 57),
            (180, 170, 148),
            (142, 150, 157),
            (62, 85, 107),
            (104, 112, 118),
        )
    elif theme == "asian":
        colors = (
            (218, 215, 203),
            (151, 108, 87),
            (196, 197, 188),
            (190, 198, 201),
            (84, 120, 132),
            (131, 143, 147),
        )
    image = Image.new("RGB", (128, 128), colors[style])
    draw = ImageDraw.Draw(image)
    if style == 1:
        for y in range(0, 128, 6):
            draw.line((0, y, 127, y), fill=(173, 116, 90))
            for x in range((y // 6 % 2) * 6, 128, 12):
                draw.line((x, y, x, y + 6), fill=(173, 116, 90))
    # Four-pixel gutters keep linear sampling inside this tile.
    for floor in range(floors):
        y0, y1 = 4 + floor * 120 / floors, 4 + (floor + 1) * 120 / floors
        if style in (2, 3):
            draw.line((4, y1 - 1, 123, y1 - 1), fill=(130, 133, 131))
        for bay in range(bays):
            x0, x1 = 4 + bay * 120 / bays, 4 + (bay + 1) * 120 / bays
            inset = 0.08 if style in (3, 4) else 0.23
            if theme == "european" and style != 4:
                inset = 0.30
            elif theme == "asian" and style in (3, 5):
                inset = 0.12
            mx = (x1 - x0) * inset
            my = (y1 - y0) * (0.08 if style == 4 else 0.20)
            shade = (bay * 13 + floor * 7 + style) % 4
            color = (45 + shade * 6, 65 + shade * 7, 76 + shade * 8)
            draw.rectangle((x0 + mx, y0 + my, x1 - mx, y1 - my), fill=color)
            if style in (0, 2) and bays <= 4 and floors <= 6:
                draw.line(((x0 + x1) / 2, y0 + my, (x0 + x1) / 2, y1 - my), fill=colors[style])
    return image


def _portuguese_facade_tile(style: int, floors: int, bays: int):
    """Original tile/plaster motifs, not reproductions of specific buildings.

    The same bounded atlas works anywhere with classified building faces.
    Shutters, balcony rails and tile joints are painted detail, not geometry.
    Tall-building families keep restrained modern window bands.
    """
    from PIL import Image, ImageDraw

    plaster = (
        (235, 213, 163),
        (211, 225, 224),
        (225, 177, 166),
        (237, 231, 214),
        (199, 214, 216),
        (223, 216, 188),
    )
    image = Image.new("RGB", (128, 128), plaster[style])
    draw = ImageDraw.Draw(image)
    if style == 1:
        # Original blue-and-white geometric azulejo-inspired repeat.
        for y in range(0, 128, 8):
            for x in range(0, 128, 8):
                draw.rectangle((x, y, x + 7, y + 7), outline=(181, 201, 204))
                draw.polygon(
                    ((x + 4, y + 1), (x + 7, y + 4), (x + 4, y + 7), (x + 1, y + 4)),
                    fill=(67, 117, 148),
                )
                draw.rectangle((x + 3, y + 3, x + 4, y + 4), fill=(231, 234, 223))
    elif style == 2:
        # Small teal tile course over warm plaster at the ground floor.
        draw.rectangle((0, 0, 127, max(4, int(120 / floors))), fill=(90, 133, 126))
    for floor in range(floors):
        y0, y1 = 4 + floor * 120 / floors, 4 + (floor + 1) * 120 / floors
        draw.line((4, y1 - 1, 123, y1 - 1), fill=(239, 235, 218))
        for bay in range(bays):
            x0, x1 = 4 + bay * 120 / bays, 4 + (bay + 1) * 120 / bays
            width, height = x1 - x0, y1 - y0
            modern = style >= 3
            mx, my = width * (0.12 if modern else 0.30), height * 0.18
            left, right, bottom, top = x0 + mx, x1 - mx, y0 + my, y1 - my
            trim = (243, 237, 220)
            draw.rectangle((left - 1, bottom - 1, right + 1, top + 1), fill=trim)
            draw.rectangle((left, bottom, right, top), fill=(54, 80, 87))
            if width >= 16 and height >= 16 and not modern:
                # Door-height openings with shutters and an iron railing.
                shutter = (54, 89, 76) if style != 2 else (102, 76, 64)
                for a, b in ((left - width * 0.13, left - 2), (right + 2, right + width * 0.13)):
                    draw.rectangle((a, bottom, b, top), fill=shutter)
                draw.line(((left + right) / 2, bottom, (left + right) / 2, top), fill=trim)
                rail_y = bottom + height * 0.12
                draw.line((left - 2, rail_y, right + 2, rail_y), fill=(53, 58, 55))
                for x in np.linspace(left - 2, right + 2, 5):
                    draw.line((x, bottom - 1, x, rail_y), fill=(53, 58, 55))
    return image


def atlas_pixels(kind: str, rgb: tuple[int, int, int], theme: str = "mixed") -> np.ndarray:
    """Generate original artwork with a fixed memory ceiling and no downloads."""
    from PIL import Image, ImageDraw

    if kind == "facades":
        image = Image.new("RGB", (2048, 2048), rgb)
        for style in range(6):
            for fi, floors in enumerate(FLOORS):
                for bi, bays in enumerate(BAYS):
                    slot = (style * len(FLOORS) + fi) * len(BAYS) + bi
                    image.paste(
                        _facade_tile(style, floors, bays, theme),
                        ((slot % 16) * 128, (slot // 16) * 128),
                    )
        draw = ImageDraw.Draw(image)
        for style in range(6):
            x, y = (240 + style) % 16 * 128, (240 + style) // 16 * 128
            shade = 75 + style * 7
            roof_color = (shade, shade + 3, shade + 5)
            if theme == "portuguese" and style < 3:
                roof_color = (164 + style * 8, 100 + style * 5, 72 + style * 4)
            elif theme == "paris":
                from .paris_facades import ROOF_COLORS

                roof_color = ROOF_COLORS[style]
            draw.rectangle((x, y, x + 127, y + 127), fill=roof_color)
        if theme == "paris":
            from .paris_facades import LIMESTONE

            # A constant stone swatch protects every excluded landmark face,
            # including roofs, without splitting the shared material mesh.
            draw.rectangle((768, 1920, 895, 2047), fill=LIMESTONE)  # Slot 246.
    else:
        image = Image.new("RGB", (1024, 3072), rgb)
        for lanes in range(1, 5):
            for oneway in (False, True):
                for li, length in enumerate(LENGTHS):
                    slot = ((lanes - 1) * 2 + int(oneway)) * len(LENGTHS) + li
                    tile = Image.new("RGB", (128, 256), (64, 67, 69))
                    draw = ImageDraw.Draw(tile)
                    # Keep junction ends clear. Lane tags control separators;
                    # generic white markings are illustrative, not local law.
                    margin = min(0.25, 5.0 / length)
                    lo, hi = int(4 + margin * 248), int(252 - margin * 248)
                    draw.line((10, lo, 10, hi), fill=(181, 184, 180), width=1)
                    draw.line((117, lo, 117, hi), fill=(181, 184, 180), width=1)
                    for lane in range(1, lanes):
                        x = round(4 + 120 * lane / lanes)
                        if not oneway and 2 * lane == lanes:
                            draw.line((x, lo, x, hi), fill=(222, 221, 204), width=2)
                        else:
                            dash = max(2, round(3 / length * 248))
                            for y in range(lo, hi, dash * 2):
                                draw.line(
                                    (x, y, x, min(y + dash, hi)), fill=(222, 221, 204), width=1
                                )
                    image.paste(tile, ((slot % 8) * 128, (slot // 8) * 256))
        for surface in range(7):
            for li, length in enumerate(LENGTHS):
                slot = 48 + surface * len(LENGTHS) + li
                tile = _surface_tile(surface, length)
                image.paste(tile, ((slot % 8) * 128, (slot // 8) * 256))
    return np.asarray(image)


def _surface_tile(surface: int, length: int):
    """Draw generic surfacing only when the source explicitly names a surface."""
    from PIL import Image, ImageDraw

    colors = (
        (159, 157, 149),
        (170, 158, 141),
        (138, 137, 131),
        (156, 143, 121),
        (124, 103, 78),
        (64, 67, 69),
        (194, 181, 146),
    )
    rng = np.random.default_rng(3700 + surface * 100 + length)
    noise = rng.integers(-9, 10, size=(256, 128, 1))
    pixels = np.clip(np.array(colors[surface])[None, None, :] + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    if surface in (0, 1, 2):
        spacing = (1.5, 0.5, 0.3)[surface]
        dy = max(2, round(248 * spacing / length))
        dx = (32, 16, 8)[surface]
        joint = tuple(max(0, c - 22) for c in colors[surface])
        for row, y in enumerate(range(4, 252, dy)):
            draw.line((4, y, 124, y), fill=joint)
            for x in range(4 - (row % 2) * (dx // 2), 125, dx):
                draw.line((x, y, x, min(252, y + dy)), fill=joint)
    return image


def _atlas(kind: str, rgb: tuple[int, int, int], theme: str) -> str:
    """Publish an atlas without invalidating another scene's material paths.

    Live consumers can belong to other scenes or processes, so a per-write
    file-count limit cannot safely evict atlases. Variants remain on disk until
    the atlas cache directory is explicitly removed; each image stays bounded.
    """
    from PIL import Image

    from ..scene.io import get_uv_cache_root

    root = get_uv_cache_root() / VERSION
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{kind}-{theme}-{'-'.join(map(str, rgb))}.png"
    if not path.is_file():
        fd, temp = tempfile.mkstemp(prefix="atlas-", suffix=".png", dir=root)
        os.close(fd)
        try:
            Image.fromarray(atlas_pixels(kind, rgb, theme)).save(temp)
            os.replace(temp, path)
        finally:
            Path(temp).unlink(missing_ok=True)
    return str(path)


def _tile_uv(slot: int, x: np.ndarray | float, y: np.ndarray | float, *, roads=False):
    columns, rows = (8, 12) if roads else (16, 16)
    # Use the same image-row convention as the existing texture loader.
    return np.stack(
        np.broadcast_arrays(
            (slot % columns + (4 + np.asarray(x) * 120) / 128) / columns,
            (
                slot // columns
                + (4 + np.asarray(y) * (248 if roads else 120)) / (256 if roads else 128)
            )
            / rows,
        ),
        axis=-1,
    )


def facade_uvs(
    source: MeshPayload, parts: list, identity: str, theme: str = "mixed", *, exclusions=()
) -> np.ndarray:
    uvs = np.empty((len(source.triangles), 3, 2), dtype=np.float32)
    uvs[:] = _tile_uv(255, 0.5, 0.5)
    excluded = set(exclusions)
    for first, count in parts:
        if (first, count) in excluded:
            uvs[first : first + count] = _tile_uv(246 if theme == "paris" else 255, 0.5, 0.5)
            continue
        corners = source.vertices[source.triangles[first : first + count]]
        minimum, maximum = corners.min(axis=(0, 1)), corners.max(axis=(0, 1))
        height = maximum[2] - minimum[2]
        seed = int.from_bytes(
            sha256(f"{identity}|{minimum.round(3)}|{maximum.round(3)}".encode()).digest()[:4],
            "little",
        )
        style = (
            (0, 1, 2)[seed % 3]
            if height < 12
            else ((3, 4, 5)[seed % 3] if height > 28 else seed % 6)
        )
        if theme in ("european", "portuguese") and height <= 28:
            style = (0, 1, 2)[seed % 3]
        elif theme == "american" and height > 18:
            style = (1, 3, 4)[seed % 3]
        elif theme == "asian" and height > 12:
            style = (2, 3, 4, 5)[seed % 4]
        fi = _nearest(FLOORS, max(1, height / (3.4 if theme == "paris" else 3)))
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        norm = np.linalg.norm(normals, axis=1)
        wall = (norm > 1e-10) & (np.abs(normals[:, 2]) <= 0.5 * norm)
        out = uvs[first : first + count]
        out[:] = _tile_uv(240 + style, 0.5, 0.5)
        ids = np.flatnonzero(wall)
        if not len(ids):
            continue
        unit = normals[ids] / norm[ids, None]
        # Coplanar faces share a projection, so triangulation adds no seams.
        keys = np.column_stack((unit, np.einsum("ij,ij->i", unit, corners[ids, 0])))
        _, groups = np.unique(keys.round(4), axis=0, return_inverse=True)
        for group in range(int(groups.max()) + 1):
            selected = ids[groups == group]
            normal = normals[selected[0]]
            tangent = np.cross((0, 0, 1), normal)
            tangent /= np.linalg.norm(tangent)
            horizontal = corners[selected] @ tangent
            width = np.ptp(horizontal)
            bi = _nearest(BAYS, max(1, width / 3))
            slot = (style * len(FLOORS) + fi) * len(BAYS) + bi
            out[selected] = _tile_uv(
                slot,
                (horizontal - horizontal.min()) / max(width, 1e-6),
                (corners[selected, :, 2] - minimum[2]) / max(height, 1e-6),
            )
    return uvs.reshape(-1, 2)


def road_uvs(source: MeshPayload, segments: list) -> np.ndarray:
    uvs = np.empty((len(source.triangles), 3, 2), dtype=np.float32)
    uvs[:] = _tile_uv(95, 0.5, 0.5, roads=True)
    for segment in segments:
        surface = SURFACES.get(str(segment.get("surface", "")).strip().lower())
        motor = segment["highway"] in _MOTOR_ROADS
        if not motor and surface is None:
            continue
        oneway = segment["oneway"]
        try:
            lanes = int(segment["lanes"])
        except (ValueError, TypeError):
            lanes = 1 if oneway or segment["highway"] in {"service", "living_street"} else 2
        lanes = min(4, max(1, lanes))
        li = _nearest(LENGTHS, segment["length_m"])
        slot = ((lanes - 1) * 2 + int(oneway)) * len(LENGTHS) + li
        if (not motor or surface in _UNPAVED) and surface is not None:
            slot = 48 + surface * len(LENGTHS) + li
        a, b = segment["start_m"] / segment["length_m"], segment["end_m"] / segment["length_m"]
        # Matches producer vertices [left-start, right-start, right-end, left-end].
        first = segment["first_triangle"]
        uvs[first : first + 2] = _tile_uv(
            slot, np.array([[0, 1, 1], [0, 1, 0]]), np.array([[a, a, b], [a, b, b]]), roads=True
        )
    return uvs.reshape(-1, 2)


def varied_payload(
    source: MeshPayload,
    metadata: list,
    identity: str,
    color,
    *,
    roads=False,
    theme="mixed",
    exclusions=(),
):
    """Resolve a cached atlas and a derived payload without changing source arrays."""
    from ..scene.io import get_uv_cache_root
    from ..scene.uv_cache_store import get_uv_cache_store

    kind = "roads" if roads else "facades"
    theme = "mixed" if roads or theme not in THEMES else theme
    digest = sha256(
        f"{VERSION}|{kind}|{theme}|{identity}|{json.dumps(metadata, sort_keys=True)}".encode()
    )
    if exclusions and not roads:
        digest.update(json.dumps(exclusions).encode())
    digest.update(np.ascontiguousarray(source.vertices).tobytes())
    digest.update(np.ascontiguousarray(source.triangles).tobytes())
    key = digest.hexdigest()
    store = get_uv_cache_store(get_uv_cache_root(), VERSION)
    uv = store.get(key, source_signature=key, expected_shape=(len(source.triangles) * 3, 2))
    if uv is None:
        uv = (
            road_uvs(source, metadata)
            if roads
            else facade_uvs(source, metadata, identity, theme, exclusions=exclusions)
        )
        store.put(key, uv, source_signature=key)
        store.flush()
    rgb = tuple(int(round(float(v) * 255)) for v in np.clip(color[:3], 0, 1))
    payload = replace(source, triangle_uvs=uv, cache_key=f"{VERSION}/{key}")
    if roads:
        from .road_surface import visible_road_payload

        payload = visible_road_payload(source, metadata, uv, cache_key=payload.cache_key)
    return payload, _atlas(kind, rgb, theme)
