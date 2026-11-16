"""Coverage-only radio annotations using the main scene's label preferences."""

from collections import Counter

import numpy as np

from ..model import RenderObject, Transform
from ..services.label_callout_service import objects_with_callouts
from ..types.label_style import LabelCalloutStyle
from ..types.render_payloads import MaterialPayload, PointCloudPayload, TextLabelPayload
from .volume_sites import transmitter_position


class CoverageSceneLabels:
    """Reuse matching actor labels; cache fallback snapshots until inputs change."""

    def __init__(self, visualizer, source):
        self.visualizer, self.source = visualizer, source
        self.key = None
        self.objects = ()
        self.active_ids = set()
        self._names = None
        self._indices = {}
        self._matches = {}
        self.positions = {
            name: transmitter_position(source, name) for name in source.get("tx_names", ())
        }

    def sync(self, *, visible=True):
        viz = self.visualizer
        state = viz.app_state
        names = tuple(getattr(state, "tx_device_names", ()) or ())
        if names != self._names:
            counts = Counter(names)
            self._indices = {name: i for i, name in enumerate(names) if counts[name] == 1}
            self._names = names
            self._matches.clear()
        enabled = visible and getattr(state, "show_labels", True)
        style = LabelCalloutStyle.from_dict(getattr(state, "label_callout", None))
        offset = tuple(
            float(getattr(viz, f"label_offset_{a}", 0.5 if a == "z" else 0)) for a in "xyz"
        )
        font_size = float(getattr(viz, "label_font_size", 0.3))
        screen_space = bool(getattr(state, "label_screen_space", True))
        positions = getattr(viz, "current_tx_positions", None)
        labels = getattr(viz, "tx_labels", ())
        entries = getattr(viz, "tx_entries", ())
        rows = []
        for name, position in self.positions.items():
            if position is None:
                continue
            index = self._indices.get(name)
            label = labels[index] if index is not None and index < len(labels) else None
            anchor = (
                np.asarray(positions[index], dtype=float)
                if index is not None and positions is not None and index < len(positions)
                else None
            )
            pose = tuple(anchor) if anchor is not None and anchor.shape == (3,) else None
            match_key = (index, pose, id(label))
            previous = self._matches.get(name)
            if previous is None or previous[0] != match_key:
                matching = (
                    label is not None
                    and pose is not None
                    and np.allclose(position, pose, rtol=0, atol=1e-4)
                )
                self._matches[name] = (match_key, bool(matching))
            matching = self._matches[name][1]
            entry = entries[index] if index is not None and index < len(entries) else {}
            allowed = entry.get("visible", True) and entry.get("show_label", True)
            selected = getattr(state, "selected_tx", "all")
            if index is not None and selected != "all" and str(index) != str(selected):
                allowed = False
            custom = getattr(state, "tx_labels", ())
            text = (
                custom[index]
                if index is not None and index < len(custom) and custom[index]
                else name
            )
            if index is not None and not matching:
                text += " (coverage)"
            rows.append((name, text, bool(matching), bool(allowed)))
        key = (enabled, style, offset, font_size, screen_space, tuple(rows))
        if key == self.key:
            return
        objects = []
        if enabled:
            for name, text, matching, allowed in rows:
                if matching or not allowed:
                    continue
                position = self.positions[name]
                prefix = f"analysis:coverage_site:{name}"
                objects.extend(
                    (
                        RenderObject(
                            id=prefix + ":point",
                            payload=PointCloudPayload(points=position[None, :]),
                            material=MaterialPayload(base_color=(1, 0, 0, 1), point_size=12),
                        ),
                        RenderObject(
                            id=prefix + ":label",
                            payload=TextLabelPayload(
                                text=text, font_size=font_size, screen_space=screen_space
                            ),
                            material=MaterialPayload(base_color=(1, 0, 0, 1)),
                            transform=Transform.from_translation(position + offset),
                            metadata=dict(
                                layout_anchor=tuple(position),
                                locator_role="radio",
                                node_kind="tx",
                                locator_companions=[prefix + ":point"],
                            ),
                        ),
                    )
                )
        decorated = tuple(objects_with_callouts(objects, viz))
        desired = {obj.id for obj in decorated}
        # Keep failed operations retryable without rebuilding the coverage layer.
        success = True
        for object_id in self.active_ids - desired:
            if viz.renderer.remove_object(object_id):
                self.active_ids.remove(object_id)
            else:
                success = False
        for obj in decorated:
            if viz.renderer.ensure_object(obj):
                self.active_ids.add(obj.id)
            else:
                success = False
        self.objects = decorated
        self.key = key if success else None

    def clear(self):
        for object_id in self.active_ids:
            self.visualizer.renderer.remove_object(object_id)
        self.active_ids.clear()
        self.objects = ()
        self.key = None
