"""pygfx label transform and layout helpers.

Pygfx text labels are native ``gfx.Text`` objects registered through the named
geometry path. This mixin lays out service-managed node/target labels so
co-located TX/RX/target labels do not occupy the same screen-space anchor.
"""

from __future__ import annotations

import numpy as np

from .callout_cards import card_padding, update_card_shape
from .callouts import position_callout, reserve_callout_rect, update_callout

__all__ = ["PygfxLabelMixin"]

_LABEL_ANCHOR_GROUP_TOLERANCE_M = 0.08
_LABEL_GROUP_VERTICAL_STEP_M = 0.34
_LABEL_GROUP_LATERAL_STEP_M = 0.18


def _background_layout_inputs(renderer, cards):
    """Capture logical inputs, never the callout positions produced by layout.

    One key covers all visible cards because collision placement is coupled.
    It retains only the current native objects and small presentation values.
    """
    result = []
    for name, (native, plate) in cards:
        options = getattr(native, "_orchav_callout_options", None)
        origin = getattr(native, "_orchav_label_connector_origin", None)
        group_key = getattr(renderer, "_label_anchor_key_by_name", {}).get(name)
        result.append(
            (
                name,
                native,
                plate,
                float(native.font_size),
                bool(native.screen_space),
                getattr(native, "_orchav_label_content", None),
                getattr(native, "_orchav_label_anchor_request", None),
                None if origin is None else tuple(origin),
                tuple(sorted(getattr(renderer, "_label_anchor_groups", {}).get(group_key, ()))),
                (
                    tuple(
                        (key, tuple(value) if isinstance(value, list) else value)
                        for key, value in sorted(options.items())
                    )
                    if options is not None
                    else native.world.matrix.tobytes()
                ),
            )
        )
    return tuple(result)


def text_label_font_size(payload) -> float:
    """Convert the neutral text scale to native pixels or world units."""
    if payload.screen_space:
        return max(10.0, min(48.0, float(payload.font_size) * 60.0))
    return max(0.05, (float(payload.font_size) / 0.3) * 0.5)


def text_label_from_payload(gfx, payload):
    """Construct the native label used by desktop and notebook adapters."""
    native = gfx.Text(
        text=payload.text,
        font_size=text_label_font_size(payload),
        screen_space=payload.screen_space,
        anchor="middle-center",
        material=gfx.TextMaterial(
            color=(1.0, 1.0, 1.0),
            outline_color=payload.outline_color,
            outline_thickness=payload.outline_thickness,
            aa=True,
        ),
    )
    native._orchav_label_content = (payload.text, False)
    return native


class PygfxLabelMixin:
    """Screen-space label layout behavior for service-owned labels.

    NodeService and TargetService own operational labels for the pygfx backend.
    These helpers maintain transform offsets for named native text labels.
    """

    @staticmethod
    def _set_label_text(native, text: str, bold: bool = False) -> None:
        """Apply glyph content only when changed, committing after success."""
        content = (text, bool(bold))
        if getattr(native, "_orchav_label_content", None) == content:
            return
        # A setter can fail after modifying native state. Retrying the old
        # requested text must repair it too, rather than trusting stale success.
        native._orchav_label_content = None
        if bold:
            native.set_markdown(f"**{text}**")
        else:
            native.set_text(text)
        native._orchav_label_content = content

    def _apply_label_metadata(self, obj, native) -> None:
        """Apply the shared text, callout, and backplate metadata."""
        native._orchav_label_card = bool(obj.metadata.get("label_card", False))
        bold = bool(obj.metadata.get("label_bold", False))
        PygfxLabelMixin._set_label_text(native, obj.payload.text, bold)
        native._orchav_label_bold = bold
        anchor_request = (
            obj.metadata.get("label_anchor", "middle-center"),
            obj.metadata.get("label_anchor_offset", 0.0),
        )
        if getattr(native, "_orchav_label_anchor_request", None) != anchor_request:
            native._orchav_label_anchor_request = None
            native.anchor, native.anchor_offset = anchor_request
            native._orchav_label_anchor_request = anchor_request
        from .callouts import sync_callout

        sync_callout(self, obj.id, obj.metadata.get("label_callout"), obj.metadata)
        PygfxLabelMixin._sync_label_background(
            self,
            obj.id,
            obj.metadata.get("label_background_rgba"),
            obj.metadata.get("label_connector_origin"),
        )

    def _sync_label_background(self, name: str, color, connector_origin=None) -> None:
        """Attach an optional native backplate; ordinary labels remain unchanged."""
        cards = getattr(self, "_label_backgrounds", None)
        native = self._objects.get(name)
        # Hidden street layers retain their native cards outside the active
        # layout registry. Reconcile that cached card before applying style
        # changes; the street synchronizer will park it again if still hidden.
        dormant = getattr(native, "_orchav_street_background", None)
        if dormant is not None:
            if cards is None:
                cards = self._label_backgrounds = {}
            cards[name] = (native, dormant)
            native._orchav_street_background = None
        if color is None and (cards is None or name not in cards):
            return
        if cards is None:
            cards = self._label_backgrounds = {}
        previous = cards.get(name)
        if color is None or native is None:
            if previous is not None:
                previous[0].remove(previous[1])
                cards.pop(name, None)
            return
        rgba = np.asarray(color, dtype=float)
        if rgba.shape != (4,) or not np.isfinite(rgba).all() or np.any((rgba < 0) | (rgba > 1)):
            raise ValueError("Label background must be an RGBA color between zero and one")
        if not native.screen_space:
            raise ValueError("Label backplates require screen-space text")
        if connector_origin is not None:
            connector_origin = np.asarray(connector_origin, dtype=float)
            if connector_origin.shape != (3,) or not np.isfinite(connector_origin).all():
                raise ValueError("Label connector origin must contain three finite coordinates")
        native._orchav_label_connector_origin = connector_origin
        if previous is None or previous[0] is not native:
            if previous is not None:
                previous[0].remove(previous[1])
            plate = self._gfx.Mesh(
                self._gfx.plane_geometry(1, 1),
                self._gfx.MeshBasicMaterial(
                    color=tuple(rgba),
                    side="both",
                    alpha_mode="blend",
                    depth_test=False,
                    depth_write=False,
                    pick_write=False,
                ),
            )
            native.add(plate)
            border = self._gfx.Line(
                self._gfx.Geometry(
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
                ),
                self._gfx.LineMaterial(
                    color=(0.2, 0.2, 0.2, 0.9),
                    thickness=1.5,
                    alpha_mode="blend",
                    depth_test=False,
                    depth_write=False,
                    pick_write=False,
                ),
            )
            plate.add(border)
            cards[name] = (native, plate)
        else:
            plate = previous[1]
            plate.material.color = tuple(rgba)
        plate.render_order = native.render_order - 2
        plate.children[0].render_order = native.render_order - 1

    def _update_label_backgrounds(self) -> None:
        """Billboard backplates at the glyphs' pixel size for drawing and export.

        Text layout supplies the bounds. Projecting the world anchor supplies
        the pixel-to-world scale, so zoom and viewport size cannot separate a
        card from its label. Cards are children and follow object visibility.
        """
        from .street_labels import update_street_labels

        update_street_labels(self)
        cards = getattr(self, "_label_backgrounds", None)
        if not cards:
            self._label_background_layout_key = None
            return
        for name, (native, plate) in tuple(cards.items()):
            if self._objects.get(name) is not native:
                cards.pop(name)
            elif not native.visible:
                plate.visible = False
        visible_cards = [(name, pair) for name, pair in sorted(cards.items()) if pair[0].visible]
        width, height = self._canvas.get_logical_size()
        if min(width, height) <= 0:
            return
        self._camera.set_view_size(width, height)
        projection = self._camera.projection_matrix
        camera_matrix = self._camera.world.matrix
        layout_key = (
            self._camera,
            width,
            height,
            projection.tobytes(),
            camera_matrix.tobytes(),
            _background_layout_inputs(self, visible_cards),
        )
        # Direct native annotations have no service-owned text/style revision.
        # Continue laying those out so native setters cannot leave stale cards.
        if any(
            getattr(native, "_orchav_label_content", None) is None
            or getattr(native, "_orchav_label_anchor_request", None) is None
            for _, (native, _) in visible_cards
        ):
            layout_key = None
        if (
            layout_key is not None
            and getattr(self, "_label_background_layout_key", None) == layout_key
        ):
            return
        # Never cache a partially applied layout, including when the next
        # request reverts to the last successful camera/actor state.
        self._label_background_layout_key = None
        view_projection = projection @ self._camera.world.inverse_matrix
        size = np.array([width, height])
        occupied = []
        # Reserve actor markers before placing cards. Label cards must not hide
        # another actor's true position, including nearby group members.
        for _, (native, _) in visible_cards:
            options = getattr(native, "_orchav_callout_options", None)
            if options is not None and native.visible:
                origin = native._orchav_label_connector_origin
                clip = view_projection @ np.r_[origin, 1]
                if clip[3] > 0:
                    pixel = (clip[:2] / clip[3] + 1) * size / 2
                    half = options["marker_size"] / 2 + 4
                    occupied.append(np.array([pixel - half, pixel + half]))
        for name, (native, plate) in visible_cards:
            request = getattr(native, "_orchav_label_anchor_request", None)
            if request is not None:
                native.anchor = request[0]
            position_callout(self, name, native, projection, camera_matrix, size, view_projection)
            anchor = native.world.matrix[:3, 3]
            clip = view_projection @ np.r_[anchor, 1]
            origin = getattr(native, "_orchav_label_connector_origin", None)
            if origin is not None and clip[3] > 0:
                origin_clip = view_projection @ np.r_[origin, 1]
                if origin_clip[3] > 0:
                    direction = (clip[:2] / clip[3] - origin_clip[:2] / origin_clip[3]) * [
                        width,
                        height,
                    ]
                    if np.linalg.norm(direction) > 1e-6:
                        if abs(direction[0]) > abs(direction[1]):
                            native.anchor = "middle-left" if direction[0] > 0 else "middle-right"
                        else:
                            native.anchor = "bottom-center" if direction[1] > 0 else "top-center"
            native._update_object()
            reserve_callout_rect(native, projection, camera_matrix, size, occupied, view_projection)
            anchor = native.world.matrix[:3, 3]
            bounds = native._aabb
            plate.visible = bool(native.visible and clip[3] > 0)
            if not plate.visible:
                continue
            units = 2 * clip[3] / (np.diag(projection)[:2] * [width, height])
            dimensions_px = bounds[1, :2] - bounds[0, :2] + 2 * card_padding(native)
            update_card_shape(self, native, plate, dimensions_px)
            dimensions = dimensions_px * units
            center = (bounds[1, :2] + bounds[0, :2]) * 0.5 * units
            world = np.eye(4)
            world[:3, :3] = camera_matrix[:3, :3] @ np.diag([*dimensions, 1])
            world[:3, 3] = anchor + camera_matrix[:3, :2] @ center
            plate.local.matrix = native.world.inverse_matrix @ world
            update_callout(self, native, plate, projection, size)
        self._label_background_layout_key = layout_key

    @staticmethod
    def _label_anchor_key(anchor: np.ndarray) -> tuple[int, int, int]:
        """Quantize nearby anchors so co-located labels share a layout group."""
        arr = np.asarray(anchor, dtype=np.float32).reshape(-1)
        if arr.size < 3:
            return (0, 0, 0)
        scale = 1.0 / _LABEL_ANCHOR_GROUP_TOLERANCE_M
        return tuple(int(round(float(v) * scale)) for v in arr[:3])

    @staticmethod
    def _label_layout_sort_key(label_key: str) -> tuple[int, str]:
        """Sort co-located labels by scene role before their stable name."""
        name = str(label_key).lower()
        if name.startswith("tx_label_") or name.startswith("node:tx_"):
            return (0, name)
        if name.startswith("rx_label_") or name.startswith("node:rx_"):
            return (1, name)
        if name.startswith("target_label_") or name.startswith("target:"):
            return (2, name)
        if name.startswith("bldg_label_") or name.startswith("scene:"):
            return (3, name)
        return (4, name)

    def _ensure_label_layout_state(self) -> None:
        """Create label layout registries on renderers restored from older state."""
        if not hasattr(self, "_label_anchor_groups"):
            self._label_anchor_groups = {}
        if not hasattr(self, "_label_anchor_key_by_name"):
            self._label_anchor_key_by_name = {}
        if not hasattr(self, "_label_anchor_by_name"):
            self._label_anchor_by_name = {}
        if not hasattr(self, "_label_offset_by_name"):
            self._label_offset_by_name = {}
        if not hasattr(self, "_label_layout_dirty_groups"):
            self._label_layout_dirty_groups = set()

    def _retry_dirty_label_layouts(self) -> bool:
        """Retry every group whose last native transform pass was incomplete."""
        dirty_groups = tuple(self._label_layout_dirty_groups)
        ok = True
        for anchor_key in dirty_groups:
            if self._layout_label_group(anchor_key):
                self._label_layout_dirty_groups.discard(anchor_key)
            else:
                ok = False
        return ok

    def _unregister_label_layout(self, label_key: str) -> None:
        """Remove one label from layout bookkeeping and relayout its old group."""
        self._ensure_label_layout_state()
        old_key = self._label_anchor_key_by_name.pop(label_key, None)
        self._label_anchor_by_name.pop(label_key, None)
        self._label_offset_by_name.pop(label_key, None)
        if old_key is None:
            return
        group = self._label_anchor_groups.get(old_key)
        if not group:
            return
        group.discard(label_key)
        if group:
            self._label_layout_dirty_groups.add(old_key)
            self._layout_label_group(old_key)
        else:
            self._label_anchor_groups.pop(old_key, None)
            self._label_layout_dirty_groups.discard(old_key)

    def _register_and_layout_label(
        self,
        label_key: str,
        anchor: np.ndarray,
        offset: np.ndarray,
    ) -> bool:
        """Register a label anchor/offset pair and recompute its anchor group."""
        self._ensure_label_layout_state()
        if not self.has_named_geometry(label_key):
            return False
        anchor3 = np.asarray(anchor, dtype=np.float32).reshape(-1)[:3]
        offset3 = np.asarray(offset, dtype=np.float32).reshape(-1)[:3]
        if anchor3.size < 3 or offset3.size < 3:
            return False
        new_key = self._label_anchor_key(anchor3)
        old_key = self._label_anchor_key_by_name.get(label_key)
        old_anchor = self._label_anchor_by_name.get(label_key)
        old_offset = self._label_offset_by_name.get(label_key)
        current_group = self._label_anchor_groups.get(new_key)
        if (
            old_key == new_key
            and old_anchor is not None
            and old_offset is not None
            and np.array_equal(old_anchor, anchor3)
            and np.array_equal(old_offset, offset3)
            and current_group is not None
            and label_key in current_group
            and self.has_named_geometry(label_key)
        ):
            return self._retry_dirty_label_layouts()
        old_group_ok = True
        if old_key is not None and old_key != new_key:
            old_group = self._label_anchor_groups.get(old_key)
            if old_group is not None:
                old_group.discard(label_key)
                if old_group:
                    self._label_layout_dirty_groups.add(old_key)
                    old_group_ok = self._layout_label_group(old_key)
                    if old_group_ok:
                        self._label_layout_dirty_groups.discard(old_key)
                else:
                    self._label_anchor_groups.pop(old_key, None)
                    self._label_layout_dirty_groups.discard(old_key)

        self._label_anchor_key_by_name[label_key] = new_key
        self._label_anchor_by_name[label_key] = anchor3.copy()
        self._label_offset_by_name[label_key] = offset3.copy()
        self._label_anchor_groups.setdefault(new_key, set()).add(label_key)
        self._label_layout_dirty_groups.add(new_key)
        new_group_ok = self._layout_label_group(new_key)
        if new_group_ok:
            self._label_layout_dirty_groups.discard(new_key)
        return old_group_ok and new_group_ok

    def _layout_label_group(self, anchor_key: tuple[int, int, int]) -> bool:
        """Spread labels attached to the same anchor in a deterministic order."""
        self._ensure_label_layout_state()
        group = self._label_anchor_groups.get(anchor_key)
        if not group:
            self._label_layout_dirty_groups.discard(anchor_key)
            return False
        labels = [
            name
            for name in sorted(group, key=self._label_layout_sort_key)
            if self.has_named_geometry(name)
        ]
        if not labels:
            self._label_anchor_groups.pop(anchor_key, None)
            self._label_layout_dirty_groups.discard(anchor_key)
            return False
        self._label_anchor_groups[anchor_key] = set(labels)
        ok_all = True
        center_index = 0.5 * float(len(labels) - 1)
        for index, name in enumerate(labels):
            anchor = self._label_anchor_by_name.get(name)
            offset = self._label_offset_by_name.get(name)
            if anchor is None or offset is None:
                ok_all = False
                continue
            layout_offset = np.asarray(offset, dtype=np.float32).copy()
            if len(labels) > 1:
                layout_offset[1] += (float(index) - center_index) * _LABEL_GROUP_LATERAL_STEP_M
                layout_offset[2] += float(index) * _LABEL_GROUP_VERTICAL_STEP_M
            if not self._apply_label_transform(name, anchor, layout_offset):
                ok_all = False
        if ok_all:
            self._label_layout_dirty_groups.discard(anchor_key)
        else:
            self._label_layout_dirty_groups.add(anchor_key)
        return ok_all

    def _apply_label_transform(
        self,
        label_key: str,
        anchor: np.ndarray,
        offset: np.ndarray,
    ) -> bool:
        """Apply the final label translation using named-geometry transforms."""
        label_pos = anchor[:3] + offset[:3]
        label_center = np.asarray(
            self._geometry_upload_center.get(label_key, np.zeros(3, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        if label_center.size < 3 or not np.all(np.isfinite(label_center[:3])):
            label_center = np.zeros(3, dtype=np.float32)
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = label_pos - label_center[:3]
        ok = self.set_named_transform(label_key, transform)
        if ok:
            self._positions[label_key] = (
                float(label_pos[0]),
                float(label_pos[1]),
                float(label_pos[2]),
            )
        return ok
