"""Deterministic Haussmann-inspired bays and optical surface maps.

Original procedural artwork, not photographs or surveyed Paris façades. One
bounded atlas is shared by the whole theme. Image row zero is architectural
bottom, matching ORCHAV's UV convention. Relief is shading, never displacement.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .surface_detail import relief_occlusion

VERSION = "haussmann-v6"
FLOORS = (3, 4, 5, 6, 7, 8)
STYLES = 3
WIDTH, HEIGHT = 256, 1024
COLUMNS, ROWS = 6, 4
GUTTER = 16
STONE_SLOT, ORIGINAL_SLOT = 21, 22
ROOF_METRES = (3.0, 12.0)
# Bake several slate repeats into one atlas tile instead of tessellating every
# repeat on large city roofs. Physical slate spacing is unchanged; only roof
# texture sampling is coarser. Facade bays retain their full texture resolution.
ROOF_REPEATS = (4, 4)
STONE = ((218, 209, 190), (225, 216, 198), (207, 203, 190))


def tile_uv(slot, x, y):
    """Map a local bay into a padded atlas tile, keeping filtering off neighbors."""
    return np.stack(
        np.broadcast_arrays(
            (slot % COLUMNS + (GUTTER + np.asarray(x) * (WIDTH - 2 * GUTTER)) / WIDTH) / COLUMNS,
            (slot // COLUMNS + (GUTTER + np.asarray(y) * (HEIGHT - 2 * GUTTER)) / HEIGHT) / ROWS,
        ),
        axis=-1,
    )


def _pbr_maps(color, relief, rough, physical_size):
    outputs = []
    for image in (color, relief, rough):
        array = np.array(image)
        array[:GUTTER] = array[GUTTER]
        array[-GUTTER:] = array[-GUTTER - 1]
        array[:, :GUTTER] = array[:, GUTTER : GUTTER + 1]
        array[:, -GUTTER:] = array[:, -GUTTER - 1 : -GUTTER]
        outputs.append(array)
    albedo, height, roughness = outputs
    height = (
        4 * height
        + np.roll(height, 1, 0)
        + np.roll(height, -1, 0)
        + np.roll(height, 1, 1)
        + np.roll(height, -1, 1)
    ) / 8
    dy, dx = np.gradient(
        height,
        physical_size[1] / (height.shape[0] - 2 * GUTTER),
        physical_size[0] / (height.shape[1] - 2 * GUTTER),
    )
    normal = np.stack((-dx, -dy, np.ones_like(dx)), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    return (
        albedo,
        np.round((normal + 1) * 127.5).astype(np.uint8),
        np.repeat(roughness[..., None], 3, axis=2),
        relief_occlusion(
            height,
            (
                physical_size[1] / (height.shape[0] - 2 * GUTTER),
                physical_size[0] / (height.shape[1] - 2 * GUTTER),
            ),
        ),
    )


def _slate_surfaces(style, size, physical_height):
    """Staggered slate courses with slight stone variation and millimetre relief."""
    width, height = size
    rng = np.random.default_rng(734 + style)
    yy, xx = np.indices((height, width))
    columns, rows = 10, max(2, 2 * round(physical_height / 0.4))
    row_coordinate = (yy - GUTTER) / (height - 2 * GUTTER) * rows
    row = np.floor(row_coordinate).astype(int)
    column_coordinate = (xx - GUTTER) / (width - 2 * GUTTER) * columns + (row % 2) * 0.5
    column = np.floor(column_coordinate).astype(int)
    values = rng.normal(0, 4.5, (rows, columns))
    variation = values[row % rows, column % columns] + rng.normal(0, 1.4, (height, width))
    # Fine mineral streaks follow the slate rather than the whole building.
    variation += 1.3 * np.sin(xx * 0.31 + np.sin(yy * 0.08))
    edge_y = row_coordinate % 1
    edge_x = column_coordinate % 1
    joint = (edge_y < 0.055) | (edge_x < 0.035)
    lip = (edge_y >= 0.055) & (edge_y < 0.12)
    variation[joint] -= 21
    variation[lip] += 11
    base = np.array((74 + style * 3, 81 + style * 3, 89 + style * 3))
    color = Image.fromarray(np.clip(base + variation[..., None], 0, 255).astype(np.uint8))
    relief = Image.fromarray(np.where(joint, -0.002, (1 - edge_y) * 0.004).astype(np.float32))
    rough = Image.fromarray(np.clip(175 + variation * 0.7, 145, 205).astype(np.uint8))
    return color, relief, rough


def _repeat_roof_tile(pixels):
    """Pack physical slate repeats into the existing bounded atlas footprint."""
    inside = pixels[GUTTER:-GUTTER, GUTTER:-GUTTER]
    width, height = WIDTH - 2 * GUTTER, HEIGHT - 2 * GUTTER
    tile = np.asarray(
        Image.fromarray(inside).resize(
            (width // ROOF_REPEATS[0], height // ROOF_REPEATS[1]), Image.Resampling.BILINEAR
        )
    )
    repeated = np.tile(tile, (ROOF_REPEATS[1], ROOF_REPEATS[0], 1))
    return np.pad(repeated, ((GUTTER, GUTTER), (GUTTER, GUTTER), (0, 0)), mode="edge")


def _bay(style, floors):
    rng = np.random.default_rng(9517 + style * 101 + floors)
    noise = rng.normal(0, 1.1, (HEIGHT, WIDTH))
    coarse = Image.fromarray(rng.integers(90, 166, (32, 8), dtype=np.uint8)).resize(
        (WIDTH, HEIGHT), Image.Resampling.BICUBIC
    )
    variation = noise + (np.asarray(coarse, dtype=float) - 128) * 0.06
    pixels = np.clip(np.array(STONE[style]) + variation[..., None], 0, 255).astype(np.uint8)
    color = Image.fromarray(pixels)
    relief = Image.new("F", (WIDTH, HEIGHT), 0.0)
    rough = Image.new("L", (WIDTH, HEIGHT), 210)
    dc, dh, dr = ImageDraw.Draw(color), ImageDraw.Draw(relief), ImageDraw.Draw(rough)
    trim = tuple(min(255, v + 9) for v in STONE[style])
    joint = tuple(v - 12 for v in STONE[style])
    frame = ((194, 190, 176), (211, 204, 188), (176, 182, 175))[style]
    iron = ((55, 59, 58), (64, 66, 61), (49, 57, 56))[style]

    def rect(box, rgb, height, roughness=205):
        dc.rectangle(box, fill=rgb)
        dh.rectangle(box, fill=float(height))
        dr.rectangle(box, fill=roughness)

    def band(y, thickness=5):
        rect((0, y, WIDTH - 1, y + thickness), trim, 0.035)
        dc.line((0, y - 1, WIDTH - 1, y - 1), fill=joint, width=1)

    # Taller shop/entry floor, repeated residential floors, and a top cornice.
    weights = np.array([1.35] + [1.0] * (floors - 1))
    levels = GUTTER + np.r_[0, np.cumsum(weights / weights.sum())] * (HEIGHT - 2 * GUTTER)
    for floor, (low, high) in enumerate(zip(levels[:-1], levels[1:], strict=True)):
        h = high - low
        attic = floor == floors - 1
        if attic:
            y0, y1 = int(low), int(high)
            for destination, patch in zip(
                (color, relief, rough), _slate_surfaces(style, (WIDTH, y1 - y0), 3.4), strict=True
            ):
                destination.paste(patch, (0, y0))
        for row, y in enumerate(np.arange(low + h / 5, high, h / 5) if not attic else []):
            dc.line((0, y, WIDTH - 1, y), fill=joint)
            dh.line((0, y, WIDTH - 1, y), fill=-0.002)
            offset = 0 if row % 2 else WIDTH // 2
            dc.line((offset, y, offset, min(y + h / 5, high)), fill=joint)
        if not attic:
            band(high - 6, 4)
        left, right = (30, 226) if floor == 0 else (64, 192)
        bottom, top = low + h * 0.12, high - h * 0.18
        if attic:
            left, right = 77, 179
            bottom, top = low + h * 0.18, high - h * 0.22
        rect((left - 8, bottom - 5, right + 8, top + 7), trim, 0.025)
        rect((left - 3, bottom - 1, right + 3, top + 2), (93, 94, 88), -0.012)
        rect((left, bottom, right, top), frame, -0.008, 150)
        # Individual panes use restrained, non-emissive sky-colored variation.
        for col in range(2):
            for row in range(3 if floor else 2):
                pane_low = bottom + 4 + row * (top - bottom - 8) / (3 if floor else 2)
                pane_high = bottom + 4 + (row + 1) * (top - bottom - 8) / (3 if floor else 2) - 3
                x0 = left + 5 + col * (right - left - 6) / 2
                x1 = left + 5 + (col + 1) * (right - left - 6) / 2 - 5
                value = 8 * ((floor + col + style) % 3)
                rect(
                    (x0, pane_low, x1, pane_high),
                    (58 + value, 76 + value, 83 + value),
                    -0.022,
                    65,
                )
                dc.line((x0 + 1, pane_high - 2, x1 - 1, pane_high - 2), fill=(108, 122, 125))
        if floor:
            # Sill and lintel, including a small central keystone.
            band_y = top + 9
            rect((left - 12, band_y, right + 12, band_y + 4), trim, 0.05)
            rect((120, top + 4, 136, top + 13), trim, 0.06)
            rect((left - 12, bottom - 9, right + 12, bottom - 5), trim, 0.045)
            balcony = floor in {2, floors - 2} and not attic
            rail_left, rail_right = (0, WIDTH - 1) if balcony else (left - 5, right + 5)
            rail_low, rail_high = bottom - 3, bottom + h * 0.22
            if balcony:
                band(rail_low - 7, 5)
            for y in (() if attic else (rail_low, rail_high)):
                rect((rail_left, y, rail_right, y + 2), iron, 0.07, 120)
            for x in (() if attic else range(int(rail_left), int(rail_right) + 1, 12)):
                rect((x, rail_low, x + 1, rail_high), iron, 0.065, 120)
                dc.ellipse((x - 3, rail_low + 4, x + 4, rail_high - 3), outline=iron, width=1)
        else:
            rect((22, high - h * 0.13, 234, high - h * 0.08), iron, 0.01, 155)
    for inset, thickness in ((8, 5), (2, 3)):
        band(levels[-2] - inset, thickness)
    return _pbr_maps(color, relief, rough, (3.0, floors * 3.4))


def atlas_arrays(rgb):
    """Return albedo, normals, roughness and cavity AO; fixed 1536×4096 size."""
    shape = (ROWS * HEIGHT, COLUMNS * WIDTH, 3)
    albedo = np.broadcast_to(np.asarray(rgb, dtype=np.uint8), shape).copy()
    normal = np.broadcast_to(np.array([128, 128, 255], dtype=np.uint8), shape).copy()
    rough = np.full(shape, 210, dtype=np.uint8)
    occlusion = np.full(shape, 255, dtype=np.uint8)
    for style in range(STYLES):
        for fi, floors in enumerate(FLOORS):
            slot = style * len(FLOORS) + fi
            x, y = slot % COLUMNS * WIDTH, slot // COLUMNS * HEIGHT
            for atlas, tile in zip(
                (albedo, normal, rough, occlusion), _bay(style, floors), strict=True
            ):
                atlas[y : y + HEIGHT, x : x + WIDTH] = tile
    for style in range(STYLES):
        slot = 18 + style
        x, y = slot % COLUMNS * WIDTH, slot // COLUMNS * HEIGHT
        tiles = _pbr_maps(*_slate_surfaces(style, (WIDTH, HEIGHT), ROOF_METRES[1]), ROOF_METRES)
        for atlas, tile in zip((albedo, normal, rough, occlusion), tiles, strict=True):
            atlas[y : y + HEIGHT, x : x + WIDTH] = _repeat_roof_tile(tile)
    x, y = STONE_SLOT % COLUMNS * WIDTH, STONE_SLOT // COLUMNS * HEIGHT
    albedo[y : y + HEIGHT, x : x + WIDTH] = STONE[0]
    return albedo, normal, rough, occlusion


def texture_paths(rgb) -> dict[str, str]:
    """Cache an atomic, portable local atlas set; generation needs no network."""
    from ..scene.io import get_uv_cache_root

    root = get_uv_cache_root() / VERSION
    root.mkdir(parents=True, exist_ok=True)
    names = ("texture_path", "normal_map_path", "roughness_map_path", "ao_map_path")
    suffix = "-".join(map(str, rgb))
    paths = {
        key: root / (f"{key}-{suffix}.png" if key == "texture_path" else f"{key}.png")
        for key in names
    }
    if not all(path.is_file() for path in paths.values()):
        for path, pixels in zip(paths.values(), atlas_arrays(rgb), strict=True):
            fd, temporary = tempfile.mkstemp(prefix="haussmann-", suffix=".png", dir=root)
            os.close(fd)
            try:
                Image.fromarray(pixels).save(temporary)
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        # Bound fallback-color variants without evicting shared detail maps.
        for stale in sorted(root.glob("texture_path-*.png"), key=lambda p: p.stat().st_mtime_ns)[
            :-32
        ]:
            stale.unlink(missing_ok=True)
    return {key: str(path) for key, path in paths.items()}
