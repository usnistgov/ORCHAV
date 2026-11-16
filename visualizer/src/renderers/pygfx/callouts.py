"""Native, screen-sized callout glyphs and guides owned by each text object."""

import numpy as np

from .callout_cards import card_padding, guide_end_at_card

_SEARCH_DIRECTIONS = ((0, 0),) + tuple(
    (dx * ring, dy * ring)
    for ring in range(1, 17)
    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1))
)


def sync_callout(renderer, name, options, metadata):
    """Create/remove one annotation group; ordinary labels allocate nothing."""
    native = renderer._objects.get(name)
    if native is None:
        return
    group = getattr(native, "_orchav_callout_group", None)
    previous = getattr(native, "_orchav_callout_options", None)
    native._orchav_callout_options = options
    if options is None:
        if group is not None:
            native.remove(group)
            native._orchav_callout_group = None
        if previous is not None:
            native.world.position = getattr(renderer, "_positions", {}).get(
                name, previous["base_position"]
            )
            native.render_order = metadata.get("render_order", 0)
            native.material.depth_test = metadata.get("depth_test", True)
            native.material.depth_write = metadata.get("depth_write", False)
        return
    gfx = renderer._gfx
    if group is None:
        group = gfx.Group()
        guide_geometry = gfx.Geometry(positions=np.zeros((2, 3), dtype=np.float32))
        for color, width in (((1, 1, 1, 1), 7), ((0, 0, 0, 1), 4)):
            group.add(
                gfx.Line(
                    guide_geometry,
                    gfx.LineSegmentMaterial(color=color, thickness=width, pick_write=False),
                )
            )
        group.add(
            gfx.Points(
                gfx.Geometry(positions=np.zeros((1, 3), dtype=np.float32)),
                gfx.PointsMarkerMaterial(
                    marker="square", edge_color="black", edge_width=2.5, pick_write=False
                ),
            )
        )
        native.add(group)
        native._orchav_callout_group = group
    for index, obj in enumerate(group.children):
        obj.material.depth_test = not options["on_top"]
        obj.material.depth_write = False
        obj.material.alpha_mode = "blend"
        obj.render_order = (31000 if options["on_top"] else 0) + (-4, -3, 10)[index]
    group.children[2].material.size = options["marker_size"]
    # Marker fill matches the label card, but stays opaque at the actual anchor.
    group.children[2].material.color = (*metadata["label_background_rgba"][:3], 1)
    card = options.get("mode") == "card"
    accent = options.get("accent_color", (0, 0, 0, 1))
    group.children[2].material.marker = "circle" if card else "square"
    group.children[2].material.edge_color = (0.97, 0.98, 1, 1) if card else (0, 0, 0, 1)
    if card:
        group.children[2].material.color = accent
    for index, line in enumerate(group.children[:2]):
        line.material.color = (
            ((0.97, 0.98, 1, 0.85) if index == 0 else accent)
            if card
            else ((1, 1, 1, 1) if index == 0 else (0, 0, 0, 1))
        )
        line.material.thickness = (5, 2.5)[index] if card else (7, 4)[index]


def position_callout(renderer, name, native, projection, camera_matrix, size, view_projection=None):
    """Move only the label in camera space; keep its marker at the actor position."""
    options = getattr(native, "_orchav_callout_options", None)
    if options is None:
        return
    base = np.asarray(options["base_position"], dtype=float)
    if view_projection is None:
        view_projection = projection @ renderer._camera.world.inverse_matrix
    clip = view_projection @ np.r_[base, 1]
    group = native._orchav_callout_group
    group.visible = bool(clip[3] > 0)
    if not group.visible:
        native.world.position = base
        return
    units = 2 * clip[3] / (np.diag(projection)[:2] * size)
    offset = np.array([options["offset_x"], options["offset_y"]], dtype=float)
    # Co-located monostatic Tx/Rx labels must remain distinguishable.
    key = getattr(renderer, "_label_anchor_key_by_name", {}).get(name)
    names = sorted(getattr(renderer, "_label_anchor_groups", {}).get(key, ()))
    if name in names:
        offset[1] += names.index(name) * (native.font_size + 24)
    native.world.position = base + camera_matrix[:3, :2] @ (offset * units)


def reserve_callout_rect(native, projection, camera_matrix, size, occupied, view_projection=None):
    """Separate cards in screen space, without moving any recorded anchor.

    The search is deterministic and bounded. Very crowded views can still
    overlap when no free screen area remains; no label is silently hidden.
    """
    if getattr(native, "_orchav_callout_options", None) is None or not native.visible:
        return
    if view_projection is None:
        view_projection = projection @ np.linalg.inv(camera_matrix)
    clip = view_projection @ np.r_[native.world.position, 1]
    if clip[3] <= 0:
        return
    pixel = (clip[:2] / clip[3] + 1) * size / 2
    bounds = native._aabb[:, :2] + pixel
    padding = card_padding(native) + 2
    bounds = bounds + np.array([-padding, padding])
    step = max(24.0, bounds[1, 1] - bounds[0, 1] + 4)
    occupied_bounds = np.asarray(occupied).reshape(-1, 2, 2)
    best, best_area = np.zeros(2), float("inf")
    for dx, dy in _SEARCH_DIRECTIONS:
        offset = (dx * step, dy * step)
        rect = bounds + offset
        outside = np.maximum(0, -rect[0]).sum() + np.maximum(0, rect[1] - size).sum()
        intersections = np.maximum(
            0,
            np.minimum(rect[1], occupied_bounds[:, 1]) - np.maximum(rect[0], occupied_bounds[:, 0]),
        )
        overlap = np.prod(intersections, axis=1).sum()
        penalty = overlap + outside * max(size)
        if penalty < best_area:
            best, best_area = np.asarray(offset), penalty
        if penalty == 0:
            break
    units = 2 * clip[3] / (np.diag(projection)[:2] * size)
    native.world.position = native.world.position + camera_matrix[:3, :2] @ (best * units)
    native._orchav_callout_rect = bounds + best
    occupied.append(bounds + best)


def update_callout(renderer, native, plate, projection, size):
    """Update fixed-pixel dashes after label layout and camera changes."""
    options = getattr(native, "_orchav_callout_options", None)
    if options is None:
        return
    group = native._orchav_callout_group
    origin = np.asarray(native._orchav_label_connector_origin, dtype=float)
    card = options.get("mode") == "card"
    end = native.world.position
    matrix = projection @ renderer._camera.world.inverse_matrix
    start_clip, end_clip = (matrix @ np.r_[p, 1] for p in (origin, end))
    group.visible = bool(start_clip[3] > 0 and end_clip[3] > 0)
    if not group.visible:
        return
    if card:
        end = guide_end_at_card(native, origin, renderer._camera, projection, size)
        end_clip = matrix @ np.r_[end, 1]
    # The group belongs to the text for lifetime/visibility, but its geometry
    # uses world coordinates so the square never follows a moved label card.
    group.local.matrix = native.world.inverse_matrix
    marker = group.children[2]
    marker_buffer = marker.geometry.positions
    marker_position = np.asarray(origin, dtype=np.float32)
    if getattr(group, "_orchav_marker_upload_pending", False) or not np.array_equal(
        marker_buffer.data[0], marker_position
    ):
        group._orchav_marker_upload_pending = True
        marker_buffer.data[:] = marker_position
        marker_buffer.update_full()
        group._orchav_marker_upload_pending = False
    length = np.linalg.norm(
        (end_clip[:2] / end_clip[3] - start_clip[:2] / start_clip[3]) * size / 2
    )
    segments = min(256, max(1, int(length / 12)))
    # Projective interpolation keeps each dash consistent along a long leader.
    fractions = np.column_stack(
        (np.arange(segments) / segments, (np.arange(segments) + 0.6) / segments)
    ).ravel()
    if card:
        fractions = np.array([0.0, 1.0])
    t = fractions * start_clip[3] / (end_clip[3] * (1 - fractions) + start_clip[3] * fractions)
    points = (origin + t[:, None] * (end - origin)).astype(np.float32)
    # The halo and dark guide share endpoints. Retain capacity across changing
    # dash counts and restrict drawing to the active prefix after shrinking.
    geometry = group.children[0].geometry
    buffer = geometry.positions
    count = len(points)
    if buffer.nitems < count:
        capacity = 1 << (count - 1).bit_length()
        data = np.broadcast_to(points[-1], (capacity, 3)).copy()
        data[:count] = points
        geometry = renderer._gfx.Geometry(positions=data)
        buffer = geometry.positions
        group._orchav_guide_upload_pending = False
    elif (
        getattr(group, "_orchav_guide_upload_pending", False)
        or buffer.draw_range != (0, count)
        or not np.array_equal(buffer.data[:count], points)
    ):
        group._orchav_guide_upload_pending = True
        buffer.data[:count] = points
        buffer.data[count:] = points[-1]
        buffer.update_full()
        group._orchav_guide_upload_pending = False
    if buffer.draw_range != (0, count):
        buffer.draw_range = (0, count)
    for line in group.children[:2]:
        if line.geometry is not geometry:
            line.geometry = geometry
        line.visible = bool(options["leader"] and length > options["marker_size"] / 2)
    plate.material.depth_test = not options["on_top"]
    border = plate.children[0]
    border.material.depth_test = not options["on_top"]
    border.material.color = (0, 0, 0, 1)
    border.material.thickness = 3.0 if card else 2.5
