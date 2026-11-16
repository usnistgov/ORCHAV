"""Scene-local display switches for geographic geometry categories."""

from __future__ import annotations

from ..scene.geographic_layers import LAYER_LABELS


class GeographicLayerService:
    """Keep category visibility independent of manual and material visibility."""

    def __init__(self, visualizer):
        self.visualizer = visualizer
        self._visible: dict[str, bool] = {}

    def clear(self) -> None:
        self._visible.clear()

    def set_scene(self) -> None:
        self.clear()
        entries = getattr(self.visualizer, "mesh_entries", [])
        self._visible = {
            layer: True
            for layer in LAYER_LABELS
            if any(entry.get("geographic_layer") == layer for entry in entries)
        }
        for entry in entries:
            entry.pop("_geographic_layer_visible", None)

    def snapshot(self) -> dict[str, bool]:
        return dict(self._visible)

    def set_visible(self, layer: str, visible: bool) -> None:
        self.restore({**self._visible, layer: visible})

    def restore(self, values: dict) -> None:
        """Restore available switches and publish changed geometry in one batch."""
        values = values if isinstance(values, dict) else {}
        changed = set()
        for layer in self._visible:
            value = values.get(layer, True)
            if type(value) is bool and self._visible[layer] != value:
                self._visible[layer] = value
                changed.add(layer)
        entries = [
            entry
            for entry in getattr(self.visualizer, "mesh_entries", [])
            if entry.get("geographic_layer") in changed
        ]
        for entry in entries:
            entry["_geographic_layer_visible"] = self._visible[entry["geographic_layer"]]
        appearance = getattr(self.visualizer, "object_appearance_service", None)
        if entries and appearance is not None:
            appearance.refresh_object_visibility_batch(entries)
        panel = getattr(getattr(self.visualizer, "ui_manager", None), "panels", {}).get("objects")
        if panel is not None:
            panel.sync_geographic_layers()
