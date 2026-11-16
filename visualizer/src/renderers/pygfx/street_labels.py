"""Bounded screen-space selection for optional street-name annotations."""

from __future__ import annotations

import numpy as np

MAX_VISIBLE_NAMES = 24
LABEL_GAP_PX = 12.0


def select_street_labels(positions, sizes, names, view_projection, viewport):
    """Choose nonoverlapping visible names, at most once per street.

    Keep annotations on their road anchors rather than moving labels onto
    neighboring streets. All positions and bounds use logical viewport pixels.
    """
    result = np.zeros(len(positions), dtype=bool)
    if not len(positions):
        return result
    clip = np.c_[positions, np.ones(len(positions))] @ view_projection.T
    with np.errstate(divide="ignore", invalid="ignore"):
        ndc = clip[:, :3] / clip[:, 3, None]
    viewport = np.asarray(viewport)
    centers = (ndc[:, :2] + 1) * viewport / 2
    half_size = np.asarray(sizes) / 2 + LABEL_GAP_PX / 2
    boxes = np.stack((centers - half_size, centers + half_size), axis=1)
    valid = (
        (clip[:, 3] > 0)
        & np.isfinite(ndc).all(axis=1)
        & (ndc[:, 2] >= -1)
        & (ndc[:, 2] <= 1)
        & (boxes[:, 0] >= 0).all(axis=1)
        & (boxes[:, 1] <= viewport).all(axis=1)
    )
    # Center-first selection remains stable during playback with a fixed camera.
    order = np.argsort(np.sum(ndc[:, :2] ** 2, axis=1), kind="stable")
    used_names, occupied = set(), []
    for index in order:
        if not valid[index] or names[index] in used_names:
            continue
        box = boxes[index]
        if any(np.all(box[0] < other[1]) and np.all(box[1] > other[0]) for other in occupied):
            continue
        result[index] = True
        occupied.append(box)
        used_names.add(names[index])
        if len(occupied) == MAX_VISIBLE_NAMES:
            break
    return result


def sync_street_label(renderer, obj, native):
    """Retain requested visibility separately from camera-dependent culling."""
    names = getattr(renderer, "_street_label_objects", None)
    street = obj.metadata.get("street_label")
    if street is not None and not obj.visible:
        # Retain native text/card resources for a cheap show/hide roundtrip,
        # but remove them from both scene traversal and per-frame card layout.
        parent = getattr(native, "parent", None)
        if parent is not None:
            native._orchav_street_parent = parent
            parent.remove(native)
        card = getattr(renderer, "_label_backgrounds", {}).pop(obj.id, None)
        if card is not None:
            native._orchav_street_background = card[1]
            renderer._label_background_layout_key = None
    else:
        parent = getattr(native, "_orchav_street_parent", None)
        if parent is not None:
            if native.parent is None:
                parent.add(native)
            native._orchav_street_parent = None
    if street is None or not obj.visible:
        if street is not None:
            native.visible = False
        if names is not None:
            names.pop(obj.id, None)
            renderer._street_label_layout_key = None
        return
    if names is None:
        names = renderer._street_label_objects = {}
    # Keep selection tied to the immutable road anchor rather than any native
    # glyph transform, so layout cannot feed back into the next selection.
    anchor = tuple(obj.transform_matrix[:3, 3])
    names[obj.id] = (native, str(street), obj.visible, anchor)
    renderer._street_label_layout_key = None
    # Both glyphs and their card obey scene depth, including road occlusion.
    card = getattr(renderer, "_label_backgrounds", {}).get(obj.id)
    if card is not None:
        for material in [card[1].material, *(child.material for child in card[1].children)]:
            material.depth_test = True


def forget_street_label(renderer, name, native):
    """Release a removed street's dormant parent/card references immediately."""
    names = getattr(renderer, "_street_label_objects", None)
    if names is not None and name in names:
        names.pop(name)
        renderer._street_label_layout_key = None
    if native is not None:
        for field in ("_orchav_street_parent", "_orchav_street_background"):
            if getattr(native, field, None) is not None:
                setattr(native, field, None)


def update_street_labels(renderer):
    """Recompute layout only when camera, viewport, or label inputs change."""
    registry = getattr(renderer, "_street_label_objects", None)
    if not registry:
        return
    for name, (native, _, _, _) in tuple(registry.items()):
        if renderer._objects.get(name) is not native:
            registry.pop(name)
    if not registry:
        renderer._street_label_layout_key = None
        return
    size = renderer._canvas.get_logical_size()
    if min(size) <= 0:
        return
    renderer._camera.set_view_size(*size)
    matrix = renderer._camera.projection_matrix @ renderer._camera.world.inverse_matrix
    inputs = tuple(
        (key, native, name, enabled, anchor, native.font_size)
        for key, (native, name, enabled, anchor) in registry.items()
    )
    key = (tuple(size), matrix.tobytes(), inputs)
    if getattr(renderer, "_street_label_layout_key", None) == key:
        return
    active = [
        (native, name, anchor) for native, name, enabled, anchor in registry.values() if enabled
    ]
    if active:
        positions = np.array([anchor for _, _, anchor in active])
        sizes = []
        for native, _, _ in active:
            native._update_object()
            sizes.append(np.ptp(native._aabb[:, :2], axis=0) + [14, 10])
        selected = select_street_labels(
            positions, sizes, [name for _, name, _ in active], matrix, size
        )
        for (native, _, _), visible in zip(active, selected):
            native.visible = bool(visible)
    for native, _, enabled, _ in registry.values():
        if not enabled:
            native.visible = False
    renderer._street_label_layout_key = key
