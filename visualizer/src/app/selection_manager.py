"""Object selection and highlight coordination for the visualizer UI.

Selection spans scene meshes plus TX/RX marker state. The manager translates
UI names to render objects, keeps selection labels current, and asks the active
renderer to refresh changed geometry through protocol-compatible helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..model import RenderObjectState
from ..scene.geometry_helpers import ensure_scene_mesh_render_state, resolve_node_label

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer


class SelectionManager:
    """Bridge selection widgets, scene entries, marker labels, and highlights."""

    def __init__(self, visualizer: OrchavVisualizer, logger: Any):
        """Store the visualizer state owner and logger used by selection actions."""
        self.visualizer = visualizer
        self.logger = logger

    def _node_label(self, kind: str, index: int) -> str:
        """Resolve a TX/RX marker label using the current app label policy."""
        app_state = getattr(self.visualizer, "app_state", None)
        is_tx = str(kind).lower() == "tx"
        return resolve_node_label(
            "TX" if is_tx else "RX",
            index,
            getattr(app_state, "tx_labels" if is_tx else "rx_labels", ()),
            label_mode=getattr(app_state, "node_label_mode", "role"),
            device_names=getattr(app_state, "tx_device_names" if is_tx else "rx_device_names", ()),
        )

    def _scene_selection_token(self, entry: dict[str, Any]) -> str:
        """Return the stable selection token for a scene entry."""
        index = next(
            (
                candidate_index
                for candidate_index, candidate in enumerate(
                    getattr(self.visualizer, "mesh_entries", [])
                )
                if candidate is entry
            ),
            None,
        )
        if entry.get("mesh") is not None:
            ensure_scene_mesh_render_state(entry, index)
        return str(entry.get("object_key") or entry.get("geometry_name") or entry["object_id"])

    def _scene_entry_for_token(self, token: Any) -> dict[str, Any] | None:
        """Resolve a scene-entry selection token back to its canonical entry."""
        if not isinstance(token, str):
            return None
        for index, entry in enumerate(getattr(self.visualizer, "mesh_entries", [])):
            if entry.get("mesh") is not None:
                ensure_scene_mesh_render_state(entry, index)
            candidates = {
                str(entry.get("object_key")),
                str(entry.get("object_id")),
                str(entry.get("geometry_name")),
            }
            if token in candidates:
                return entry
        return None

    def _scene_entry_for_geometry(self, geometry: Any) -> dict[str, Any] | None:
        """Return the scene entry that owns a geometry object by identity."""
        for entry in getattr(self.visualizer, "mesh_entries", []):
            if entry.get("mesh") is geometry:
                return entry
        return None

    def _set_scene_entry_highlight(self, entry: dict[str, Any], highlighted: bool) -> None:
        """Republish selection-derived highlight without changing manual state."""
        del highlighted
        appearance = getattr(self.visualizer, "object_appearance_service", None)
        refresh = getattr(appearance, "refresh_entry_material", None)
        if callable(refresh):
            refresh(entry)

    def populate_dropdown(self) -> None:
        """Populate the visible-building dropdown using the active search text."""
        viz = self.visualizer
        dropdown = getattr(viz, "object_selection_dropdown", None)
        if dropdown is None:
            return

        dropdown.clear()
        dropdown.addItem("Select a building...")
        search_text = (
            viz.object_search_box.text().lower() if hasattr(viz, "object_search_box") else ""
        )

        count = 0
        for entry in viz.mesh_entries:
            if entry.get("visible", True):
                if not search_text or search_text in entry["name"].lower():
                    dropdown.addItem(entry["name"])
                    count += 1

        self.logger.info("Selection dropdown populated: %s buildings", count)
        label = getattr(viz, "selection_info_label", None)
        if label is None:
            return
        if count == 0 and search_text:
            label.setText("No buildings match search criteria")
        else:
            self.update_selection_info()

    def update_selection_info(self) -> None:
        """Refresh the selected-object label with scene and TX/RX names."""
        viz = self.visualizer
        label = getattr(viz, "selection_info_label", None)
        if label is None:
            return
        if not getattr(viz, "selected_objects", None):
            label.setText("No objects selected")
            return

        selected_names = []
        for obj in viz.selected_objects:
            name = "Unknown"
            scene_entry = self._scene_entry_for_token(obj) or self._scene_entry_for_geometry(obj)
            if scene_entry is not None:
                name = scene_entry["name"]
                selected_names.append(name)
                continue
            for i, marker in enumerate(getattr(viz, "tx_markers", [])):
                if marker == obj:
                    name = self._node_label("tx", i)
                    break
            for i, marker in enumerate(getattr(viz, "rx_markers", [])):
                if marker == obj:
                    name = self._node_label("rx", i)
                    break
            selected_names.append(name)

        label.setText(
            "Selected: " + ", ".join(selected_names[:5]) + ("…" if len(selected_names) > 5 else "")
        )

    def pick_object_by_name(self, object_name: str) -> None:
        """Resolve a UI object name and toggle the matching scene item."""
        viz = self.visualizer
        pickable_objects = []

        for entry in viz.mesh_entries:
            if entry.get("visible", True):
                pickable_objects.append(
                    {
                        "geometry": entry["mesh"],
                        "selection_token": self._scene_selection_token(entry),
                        "type": "building",
                        "name": entry["name"],
                        "entry": entry,
                    }
                )

        for i, marker in enumerate(viz.tx_markers):
            pickable_objects.append(
                {"geometry": marker, "type": "tx", "name": self._node_label("tx", i), "index": i}
            )

        for i, marker in enumerate(viz.rx_markers):
            pickable_objects.append(
                {"geometry": marker, "type": "rx", "name": self._node_label("rx", i), "index": i}
            )

        for obj in pickable_objects:
            if obj["name"] == object_name:
                self.toggle_object_selection(obj)
                self.logger.info(f"Picked object by name: {object_name}")
                break
        else:
            self.logger.warning(f"Object not found: {object_name}")

    def toggle_object_selection(self, obj_info: dict[str, Any]) -> None:
        """Toggle selection state for the provided object."""
        viz = self.visualizer
        geometry = obj_info["geometry"]
        selection_token = obj_info.get("selection_token", geometry)
        selected_objects = getattr(viz, "selected_objects", None)
        if selected_objects is None:
            selected_objects = set()
            viz.selected_objects = selected_objects

        if selection_token in selected_objects:
            selected_objects.discard(selection_token)
            if obj_info.get("type") == "building" and isinstance(obj_info.get("entry"), dict):
                self._set_scene_entry_highlight(obj_info["entry"], False)
            else:
                self.remove_selection_highlight(geometry)
            self.logger.info(f"Deselected: {obj_info['name']}")
        else:
            selected_objects.add(selection_token)
            if obj_info.get("type") == "building" and isinstance(obj_info.get("entry"), dict):
                self._set_scene_entry_highlight(obj_info["entry"], True)
            else:
                self.add_selection_highlight(geometry)
            self.logger.info(f"Selected: {obj_info['name']} ({obj_info['type']})")
            ui = getattr(viz, "ui_controller", None)
            if ui is not None:
                ui.handle_object_selected(obj_info)

        self.update_selection_info()

    def add_selection_highlight(self, geometry: Any) -> None:
        """Publish transient highlight without changing neutral geometry colors."""
        scene_entry = self._scene_entry_for_geometry(geometry)
        if scene_entry is not None:
            self._set_scene_entry_highlight(scene_entry, True)
            return
        appearance = getattr(self.visualizer, "object_appearance_service", None)
        refresh = getattr(appearance, "refresh_transient_selection_object", None)
        if callable(refresh) and isinstance(geometry, RenderObjectState):
            refresh(geometry, selected=True)

    def remove_selection_highlight(self, geometry: Any) -> None:
        """Restore a transient selection snapshot from canonical material state."""
        scene_entry = self._scene_entry_for_geometry(geometry)
        if scene_entry is not None:
            self._set_scene_entry_highlight(scene_entry, False)
            return
        appearance = getattr(self.visualizer, "object_appearance_service", None)
        refresh = getattr(appearance, "refresh_transient_selection_object", None)
        if callable(refresh) and isinstance(geometry, RenderObjectState):
            refresh(geometry, selected=False)

    def clear_selections(self) -> None:
        """Remove all selections from the scene."""
        viz = self.visualizer
        selected_objects = list(getattr(viz, "selected_objects", []))
        viz.selected_objects.clear()
        for selected in selected_objects:
            scene_entry = self._scene_entry_for_token(selected)
            if scene_entry is not None:
                self._set_scene_entry_highlight(scene_entry, False)
            else:
                self.remove_selection_highlight(selected)
        self.logger.info("All selections cleared")
        self.update_selection_info()
