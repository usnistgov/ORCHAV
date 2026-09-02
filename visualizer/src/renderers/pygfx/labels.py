"""pygfx label transform and layout helpers.

Pygfx text labels are native ``gfx.Text`` objects registered through the named
geometry path. This mixin lays out service-managed node/target labels so
co-located TX/RX/target labels do not occupy the same screen-space anchor.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PygfxLabelMixin"]

_LABEL_ANCHOR_GROUP_TOLERANCE_M = 0.08
_LABEL_GROUP_VERTICAL_STEP_M = 0.34
_LABEL_GROUP_LATERAL_STEP_M = 0.18


class PygfxLabelMixin:
    """Screen-space label layout behavior for service-owned labels.

    NodeService and TargetService own operational labels for the pygfx backend.
    These helpers maintain transform offsets for named native text labels.
    """

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
