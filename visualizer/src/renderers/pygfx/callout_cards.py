"""Padded, rounded native backplates for the optional Card label style."""

import numpy as np


def card_padding(native):
    """Return per-side pixel padding; legacy backplates retain their 5 px inset."""
    if getattr(native, "_orchav_label_card", False):
        return np.array([4.0, 3.0])
    options = getattr(native, "_orchav_callout_options", None) or {}
    if options.get("mode") != "card":
        return np.array([5.0, 5.0])
    return np.array([max(12.0, native.font_size * 0.32), max(8.0, native.font_size * 0.18)])


def update_card_shape(renderer, native, plate, dimensions_px):
    """Cache a rounded fan and border in the backplate's normalized plane."""
    options = getattr(native, "_orchav_callout_options", None) or {}
    rounded = bool(getattr(native, "_orchav_label_card", False)) or options.get("mode") == "card"
    key = (rounded, *dimensions_px, float(native.font_size))
    if getattr(plate, "_orchav_shape_key", None) == key:
        return
    previous = getattr(plate, "_orchav_shape_key", None)
    if not rounded:
        if previous is not None and previous[0]:
            plate.geometry = renderer._gfx.plane_geometry(1, 1)
            plate.children[0].geometry = renderer._gfx.Geometry(
                positions=np.array(
                    [
                        [-0.5, -0.5, 0],
                        [0.5, -0.5, 0],
                        [0.5, 0.5, 0],
                        [-0.5, 0.5, 0],
                        [-0.5, -0.5, 0],
                    ],
                    dtype=np.float32,
                )
            )
        plate._orchav_shape_key = key
        return
    width, height = dimensions_px
    radius = min(max(7.0, native.font_size * 0.16), width / 2, height / 2)
    perimeter = []
    for cx, cy, start in (
        (width / 2 - radius, height / 2 - radius, 0),
        (-width / 2 + radius, height / 2 - radius, 90),
        (-width / 2 + radius, -height / 2 + radius, 180),
        (width / 2 - radius, -height / 2 + radius, 270),
    ):
        angles = np.deg2rad(np.linspace(start, start + 90, 9))
        perimeter.extend(zip(cx + radius * np.cos(angles), cy + radius * np.sin(angles)))
    points = np.column_stack([np.asarray(perimeter) / [width, height], np.zeros(len(perimeter))])
    vertices = np.vstack([[0, 0, 0], points]).astype(np.float32)
    indices = np.array(
        [[0, i + 1, (i + 1) % len(points) + 1] for i in range(len(points))], dtype=np.uint32
    )
    plate.geometry = renderer._gfx.Geometry(positions=vertices, indices=indices)
    plate.children[0].geometry = renderer._gfx.Geometry(
        positions=np.vstack([points, points[0]]).astype(np.float32)
    )
    plate._orchav_shape_key = key


def guide_end_at_card(native, origin, camera, projection, size):
    """Stop the guide at the card boundary, outside the text padding."""
    bounds = native._orchav_callout_rect
    matrix = projection @ camera.world.inverse_matrix
    clip = matrix @ np.r_[origin, 1]
    origin_px = (clip[:2] / clip[3] + 1) * size / 2
    end_px = np.clip(origin_px, bounds[0] + 2, bounds[1] - 2)
    anchor = native.world.position
    anchor_clip = matrix @ np.r_[anchor, 1]
    anchor_px = (anchor_clip[:2] / anchor_clip[3] + 1) * size / 2
    units = 2 * anchor_clip[3] / (np.diag(projection)[:2] * size)
    end = anchor + camera.world.matrix[:3, :2] @ ((end_px - anchor_px) * units)
    return end
