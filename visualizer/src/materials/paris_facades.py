"""Restrained, illustrative limestone facades in the existing bounded atlas.

All detail is original raster artwork. It adds no geometry, texture samplers,
or draw calls, and does not assert surveyed Paris facade or roof materials.
"""

from PIL import Image, ImageDraw

LIMESTONE = (216, 208, 190)
ROOF_COLORS = (
    (105, 115, 123),
    (114, 123, 130),
    (97, 107, 117),
    (119, 125, 129),
    (108, 117, 124),
    (124, 129, 132),
)
_STONE_COLORS = (
    (219, 211, 194),
    (211, 204, 189),
    (226, 218, 201),
    (216, 211, 198),
    (221, 216, 204),
    (207, 202, 189),
)


def facade_tile(style: int, floors: int, bays: int) -> Image.Image:
    """Paint narrow windows, stone courses and cornices at bounded detail."""
    stone = _STONE_COLORS[style]
    trim = tuple(min(255, value + 10) for value in stone)
    joint = tuple(value - 14 for value in stone)
    image = Image.new("RGB", (128, 128), stone)
    draw = ImageDraw.Draw(image)
    for floor in range(floors):
        y0, y1 = 4 + floor * 120 / floors, 4 + (floor + 1) * 120 / floors
        height = y1 - y0
        if height >= 15:
            for fraction in (0.25, 0.5, 0.75):
                y = y0 + height * fraction
                draw.line((4, y, 123, y), fill=joint)
        draw.line((4, y1 - 1, 123, y1 - 1), fill=trim)
        for bay in range(bays):
            x0, x1 = 4 + bay * 120 / bays, 4 + (bay + 1) * 120 / bays
            width = x1 - x0
            left, right = x0 + width * 0.29, x1 - width * 0.29
            bottom, top = y0 + height * 0.18, y1 - height * 0.16
            draw.rectangle((left - 1, bottom - 1, right + 1, top + 1), fill=trim)
            shade = (bay + 2 * floor + style) % 3
            draw.rectangle(
                (left, bottom, right, top),
                fill=(66 + shade * 4, 77 + shade * 4, 81 + shade * 4),
            )
            if width >= 20 and height >= 15:
                center = (left + right) / 2
                draw.line((center, bottom, center, top), fill=trim)
                draw.line((left, bottom + height * 0.44, right, bottom + height * 0.44), fill=trim)
                # Painted rail; no displacement or implied balcony geometry.
                draw.line((left - 1, top - 1, right + 1, top - 1), fill=(80, 82, 79))
    return image
