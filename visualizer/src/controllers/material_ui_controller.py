"""Material and object-display UI controller.

``MaterialUIController`` turns material-panel actions into scene entry updates,
PBR override changes, renderer material refreshes, and MPC material-filter UI
state. Material services own material IDs, modes, and PBR policy; this
controller owns user interaction glue, cache invalidation, and small Qt display
widgets such as the colorbar and object list.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ..materials.texture_policy import resolve_texture_policy
from ..services.cache_service import CacheInvalidationScope, invalidate_visualizer_cache

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer
    from ..services.material_entry_editor import MaterialEntryEditService
    from ..services.material_mode_commands import MaterialModeCommandService

logger = logging.getLogger(__name__)


class MaterialUIController:
    """Bridge material UI controls to services, caches, and renderer refreshes."""

    def __init__(self, parent: Any) -> None:
        """Store the parent UI controller that owns visualizer services."""
        self._parent = parent

    @property
    def visualizer(self) -> OrchavVisualizer:
        """Return the visualizer owned by the parent UI controller."""
        return self._parent.visualizer

    @property
    def material_mode_command_service(self) -> Optional[MaterialModeCommandService]:
        """Return the material-mode command service owned by the parent controller."""
        return self._parent.material_mode_command_service

    @property
    def material_entry_edit_service(self) -> Optional[MaterialEntryEditService]:
        """Return the material entry-edit service owned by the parent UI controller."""
        return self._parent.material_entry_edit_service

    def _invalidate_material_caches(self, *, reason: str) -> None:
        """Invalidate material-derived ViewModel/canonical color caches."""
        invalidate_visualizer_cache(
            self.visualizer,
            CacheInvalidationScope.MATERIALS_COLORS,
            reason=reason,
        )

    def handle_material_color_changed(self, entry: Dict[str, Any]) -> None:
        """Handle visual-material color changes from the color picker.

        Explicit visual assignments remain independent of raw scene
        ``material_type`` changes, so edits use the entry's stable visual key.
        Raw scene ``material_id`` changes use ``handle_material_id_changed``
        instead.
        """
        viz = self.visualizer
        if not hasattr(viz, "dialog_manager"):
            logger.warning("DialogManager not available for color picker")
            return
        pbr_service = getattr(viz, "material_pbr_service", None)
        if pbr_service is None:
            logger.warning("MaterialPBRService not available for appearance color editing")
            return

        get_visual_key = getattr(pbr_service, "get_visual_material_key", None)
        material_key = (
            str(get_visual_key(entry))
            if callable(get_visual_key)
            else str(entry.get("material_type") or "")
        )
        if not material_key:
            logger.warning(
                "Cannot edit appearance color for '%s': missing visual material key",
                entry.get("name", "Unknown"),
            )
            return

        resolve_material = getattr(pbr_service, "resolve_entry_material", None)
        if callable(resolve_material):
            resolved_material = resolve_material(entry)
            effective = resolved_material.properties_copy()
            texture_policy = resolved_material.texture_policy
        else:
            effective = pbr_service.get_effective_entry_properties(entry)
            texture_policy = resolve_texture_policy(
                effective,
                color=effective.get("color", entry.get("color", [0.7, 0.7, 0.7])),
                alpha=effective.get("alpha", 1.0),
                context=material_key,
            )
        if not texture_policy.color_editable:
            # Active albedo textures own base color; editing the scalar color
            # would be hidden by the renderer texture policy.
            logger.info(
                "Skipping appearance color edit for '%s': active albedo texture controls color",
                material_key,
            )
            return
        init_color = effective.get("color", entry.get("color", [0.7, 0.7, 0.7]))
        init = QColor(*[int(c * 255) for c in init_color[:3]])
        color = viz.dialog_manager.pick_color(init, "Pick Visual Material Color")
        if not color or not color.isValid():
            return

        rgb = [color.redF(), color.greenF(), color.blueF()]
        if not pbr_service.set_property(material_key, "color", rgb):
            return

        self._parent.populate_controls()

    def handle_material_id_changed(self, entry: Dict[str, Any], new_id: str) -> None:
        """Assign a scene material ID and refresh affected renderer entries."""
        if not self.material_entry_edit_service:
            logger.warning("MaterialEntryEditService not available for material ID editing")
            return

        viz = self.visualizer
        if not new_id:
            return

        mpc_core = getattr(viz, "mpc_core", None)

        old_material_id = entry.get("material_id", "")
        actual_material_id, _new_color, _new_bsdf = (
            self.material_entry_edit_service.change_material_id(
                entry,
                new_id,
                getattr(viz, "xml_root", None),
                viz.mesh_entries,
                viz.target_entries,
                mpc_core=mpc_core,
            )
        )
        if actual_material_id == old_material_id:
            return

        if hasattr(viz, "vis_initialized") and viz.vis_initialized:
            updated_entries = self._matching_entries_after_material_id_change(entry)
            appearance = getattr(viz, "object_appearance_service", None)
            refresh_batch = getattr(appearance, "refresh_entry_appearance_batch", None)
            if callable(refresh_batch):
                refresh_batch(updated_entries)

            # Recreate target label if this entry is a target (label color is baked
            # into vertex data, so it must be rebuilt when material changes).
            if any(e.get("entry_type") == "target" for e in updated_entries):
                viz.node_service.recreate_target_labels(getattr(viz, "label_font_size", 0.3))

        self._invalidate_material_caches(reason="material_id")
        if hasattr(viz, "schedule_update"):
            viz.schedule_update()

        scene_edit = getattr(viz, "scene_edit_service", None)
        if scene_edit is not None:
            scene_edit.push_scene_updates_online(
                f"Material update '{entry.get('name', 'Unknown')}'"
            )

        self._parent.populate_controls()

    def apply_material_modes(self, material_key: str | None = None) -> None:
        """Apply material show/hide/highlight modes to scene and target entries."""
        if not self.material_mode_command_service:
            logger.warning("MaterialModeCommandService not available for material modes")
            return

        viz = self.visualizer
        appearance = getattr(viz, "object_appearance_service", None)
        if appearance is None:
            logger.warning("ObjectAppearanceService not available for material filtering")
            return

        self.material_mode_command_service.apply_material_modes(
            getattr(viz, "mesh_entries", []),
            getattr(viz, "target_entries", []),
            appearance.refresh_entry_appearance_batch,
            material_key=material_key,
            visual_material_key=getattr(
                getattr(viz, "material_pbr_service", None),
                "get_visual_material_key",
                None,
            ),
            update_renderer=True,
        )

    def _matching_entries_after_material_id_change(
        self, entry: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Return scene/target entries sharing the edited material identity."""
        viz = self.visualizer
        entry_name = entry.get("name")
        entry_type = entry.get("entry_type", "mesh")
        candidates = (
            getattr(viz, "target_entries", [])
            if entry_type == "target"
            else getattr(viz, "mesh_entries", [])
        )
        matches: List[Dict[str, Any]] = []
        for candidate in candidates:
            if candidate is entry or (entry_name and candidate.get("name") == entry_name):
                matches.append(candidate)
        if entry not in matches:
            matches.append(entry)
        return matches

    def update_colorbar(self, title: str, value_range: List[float]) -> None:
        """Render the floating colorbar for the active MPC color mode."""
        viz = self.visualizer
        if viz.colorbar_widget is None:
            logger.debug("Colorbar widget is None!")
            return

        logger.debug("Updating colorbar: %s with range %s", title, value_range)

        if hasattr(viz.colorbar_widget, "layout") and viz.colorbar_widget.layout() is not None:
            layout = viz.colorbar_widget.layout()
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        else:
            layout = QVBoxLayout(viz.colorbar_widget)
            layout.setContentsMargins(15, 8, 15, 8)
            layout.setSpacing(8)

        title_label = QLabel(title)
        title_label.setStyleSheet(
            "color: #000000; font-weight: bold; font-size: 18pt; padding: 8px;"
            " border-radius: 3px; border: 1px solid #f0f0f0; background-color: #f0f0f0;"
        )
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setMinimumHeight(55)
        title_label.setMaximumHeight(60)
        layout.addWidget(title_label)

        gradient_widget = QWidget()
        gradient_widget.setMinimumHeight(35)
        gradient_widget.setMaximumHeight(40)

        gradient_css = self._generate_colormap_gradient_css()
        gradient_widget.setStyleSheet(f"""
            QWidget {{
                background: {gradient_css};
                border: 3px solid #888888;
                border-radius: 8px;
                margin: 3px;
            }}
        """)
        layout.addWidget(gradient_widget)

        range_layout = QHBoxLayout()

        # Delay modes are displayed in nanoseconds; other color modes use the
        # already-normalized unit labels in their titles.
        if "Delay" in title:
            min_text = f"{value_range[0]:.2f} ns"
            max_text = f"{value_range[1]:.2f} ns"
        else:
            min_text = f"{value_range[0]:.1f}"
            max_text = f"{value_range[1]:.1f}"

        min_color, max_color = self._get_colormap_endpoint_colors()

        min_label = QLabel(min_text)
        min_label.setStyleSheet(
            f"color: {min_color}; font-size: 11pt; font-weight: bold;"
            " border: 1px solid #f0f0f0; background-color: #f0f0f0; padding: 5px;"
        )
        min_label.setMinimumHeight(35)
        min_label.setMaximumHeight(40)
        min_label.setMinimumWidth(160)
        min_label.setMaximumWidth(160)
        min_label.setAlignment(Qt.AlignCenter)
        min_label.setWordWrap(False)
        range_layout.addWidget(min_label)

        range_layout.addStretch()

        max_label = QLabel(max_text)
        max_label.setStyleSheet(
            f"color: {max_color}; font-size: 11pt; font-weight: bold;"
            " border: 1px solid #f0f0f0; background-color: #f0f0f0; padding: 5px;"
        )
        max_label.setMinimumHeight(35)
        max_label.setMaximumHeight(40)
        max_label.setMinimumWidth(160)
        max_label.setMaximumWidth(160)
        max_label.setAlignment(Qt.AlignCenter)
        max_label.setWordWrap(False)
        range_layout.addWidget(max_label)

        layout.addLayout(range_layout)

        viz.colorbar_widget.setMinimumHeight(120)
        viz.colorbar_widget.setMaximumHeight(150)
        viz.colorbar_widget.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                border: 2px solid #cccccc;
                border-radius: 6px;
                margin: 3px;
                padding: 6px;
            }
        """)

        viz.colorbar_widget.show()
        viz.colorbar_widget.raise_()
        viz.colorbar_widget.repaint()
        viz.colorbar_widget.update()

        if viz.colorbar_widget.parent():
            viz.colorbar_widget.parent().updateGeometry()

        logger.debug("Colorbar updated successfully: %s with range %s", title, value_range)

    def _generate_colormap_gradient_css(self) -> str:
        """Generate Qt gradient CSS from the active statistics colormap."""
        try:
            from shared.statistics.themes import theme_manager

            cmap_name = theme_manager.current.continuous_colormap

            try:
                import matplotlib as mpl

                if hasattr(mpl, "colormaps"):
                    cmap = mpl.colormaps.get_cmap(cmap_name)
                else:
                    import matplotlib.pyplot as plt

                    cmap = plt.cm.get_cmap(cmap_name)

                stops = []
                for i, pos in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
                    rgba = cmap(pos)
                    r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
                    stops.append(f"stop:{pos} #{r:02x}{g:02x}{b:02x}")

                return f"qlineargradient(x1:0, y1:0, x2:1, y2:0, {', '.join(stops)})"

            except (ValueError, AttributeError) as e:
                logger.debug("Failed to load colormap %s: %s", cmap_name, e)

        except (ImportError, AttributeError) as e:
            logger.debug("Failed to get theme colormap: %s", e)

        return (
            "qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            " stop:0 #00FF88, stop:0.5 #FFD700, stop:1 #FF4444)"
        )

    def _get_colormap_endpoint_colors(self) -> tuple[str, str]:
        """Return min/max colors from the active statistics colormap."""
        try:
            from shared.statistics.themes import theme_manager

            cmap_name = theme_manager.current.continuous_colormap

            try:
                import matplotlib as mpl

                if hasattr(mpl, "colormaps"):
                    cmap = mpl.colormaps.get_cmap(cmap_name)
                else:
                    import matplotlib.pyplot as plt

                    cmap = plt.cm.get_cmap(cmap_name)

                min_rgba = cmap(0.0)
                max_rgba = cmap(1.0)

                min_hex = (
                    f"#{int(min_rgba[0]*255):02x}"
                    f"{int(min_rgba[1]*255):02x}"
                    f"{int(min_rgba[2]*255):02x}"
                )
                max_hex = (
                    f"#{int(max_rgba[0]*255):02x}"
                    f"{int(max_rgba[1]*255):02x}"
                    f"{int(max_rgba[2]*255):02x}"
                )

                return min_hex, max_hex

            except (ValueError, AttributeError) as e:
                logger.debug("Failed to get colormap endpoint colors: %s", e)

        except (ImportError, AttributeError) as e:
            logger.debug("Failed to get theme colormap: %s", e)

        return "#006600", "#cc0000"

    def update_object_list_display(self, mode: str) -> None:
        """Rebuild the object-list widget in the requested grouping mode."""
        viz = self.visualizer
        panel = getattr(getattr(viz, "ui_manager", None), "panels", {}).get("objects")
        if panel is not None and hasattr(panel, "apply_grouping"):
            mode_map = {
                "Alphabetical": "Alphabetical",
                "By Material": "Material ID",
                "Material": "Material ID",
                "Material ID": "Material ID",
                "By Type": "Type",
                "Type": "Type",
                "Compact": "None",
            }
            panel.apply_grouping(mode_map.get(mode, mode or "Material ID"))
            return

        if not hasattr(viz, "object_list_layout"):
            return

        for i in reversed(range(viz.object_list_layout.count())):
            child = viz.object_list_layout.itemAt(i).widget()
            if child:
                child.deleteLater()

        if mode == "Alphabetical":
            self._display_objects_alphabetically()
        elif mode == "By Material":
            self._display_objects_by_material()
        elif mode == "By Type":
            self._display_objects_by_type()
        elif mode == "Compact":
            self._display_objects_compact()

    def _display_objects_alphabetically(self) -> None:
        """Display objects in alphabetical order."""
        viz = self.visualizer
        object_names = []
        for entry in viz.mesh_entries:
            if entry.get("visible", True):
                object_names.append(entry["name"])

        object_names.sort()

        for name in object_names:
            label = QLabel(f"{name}")
            label.setStyleSheet("margin: 2px; padding: 2px;")
            viz.object_list_layout.addWidget(label)

    def _display_objects_by_material(self) -> None:
        """Display visible scene entries grouped by their display material key."""
        viz = self.visualizer
        material_groups: Dict[str, List[str]] = {}
        for entry in viz.mesh_entries:
            if entry.get("visible", True):
                material = self._entry_material_label(entry)
                if material not in material_groups:
                    material_groups[material] = []
                material_groups[material].append(entry["name"])

        for material, names in material_groups.items():
            header = QLabel(f"{material}")
            header.setStyleSheet("font-weight: bold; margin: 5px 2px 2px 2px; color: #333;")
            viz.object_list_layout.addWidget(header)

            for name in sorted(names):
                label = QLabel(f"  - {name}")
                label.setStyleSheet("margin: 2px 2px 2px 20px;")
                viz.object_list_layout.addWidget(label)

    def _display_objects_by_type(self) -> None:
        """Display visible scene entries grouped by structured object type."""
        viz = self.visualizer
        categorized: Dict[str, List[str]] = {}
        for entry in viz.mesh_entries:
            if entry.get("visible", True):
                category = self._entry_type_label(entry, default_type="mesh")
                categorized.setdefault(category, []).append(entry["name"])

        for category, names in categorized.items():
            if names:
                header = QLabel(category)
                header.setStyleSheet("font-weight: bold; margin: 5px 2px 2px 2px; color: #333;")
                viz.object_list_layout.addWidget(header)

                for name in sorted(names):
                    label = QLabel(f"  - {name}")
                    label.setStyleSheet("margin: 2px 2px 2px 20px;")
                    viz.object_list_layout.addWidget(label)

    def _display_objects_compact(self) -> None:
        """Display objects in a compact columnar format."""
        viz = self.visualizer
        object_names = []
        for entry in viz.mesh_entries:
            if entry.get("visible", True):
                object_names.append(entry["name"])

        object_names.sort()

        compact_layout = QHBoxLayout()

        col_size = max(1, len(object_names) // 3 + 1)
        for i in range(0, len(object_names), col_size):
            col_layout = QVBoxLayout()
            col_names = object_names[i : i + col_size]

            for name in col_names:
                label = QLabel(f"{name}")
                label.setStyleSheet("margin: 1px; padding: 1px; font-size: 10px;")
                col_layout.addWidget(label)

            compact_layout.addLayout(col_layout)

        viz.object_list_layout.addLayout(compact_layout)

    @staticmethod
    def _entry_material_label(entry: Dict[str, Any]) -> str:
        """Return the best current-schema material label for object lists."""
        for key in ("material_id", "material_type", "material"):
            value = entry.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text and text.lower() != "unknown":
                return text
        return "Unknown"

    @classmethod
    def _entry_type_label(cls, entry: Dict[str, Any], *, default_type: str = "mesh") -> str:
        """Return a structured type group label without guessing from names."""
        entry_type = str(entry.get("entry_type") or default_type or "mesh").strip().lower()
        fixed_labels = {
            "mesh": "Scene Meshes",
            "scene": "Scene Meshes",
            "scene_mesh": "Scene Meshes",
            "target": "Targets",
            "tx": "Transmitters",
            "rx": "Receivers",
        }
        if entry_type in fixed_labels:
            if entry_type in {"mesh", "scene", "scene_mesh"}:
                structured = entry.get("object_type") or entry.get("type") or entry.get("category")
                if structured:
                    return cls._format_group_label(str(structured))
            return fixed_labels[entry_type]
        return cls._format_group_label(entry_type)

    @staticmethod
    def _format_group_label(value: str) -> str:
        """Format a structured type/category value for display."""
        text = value.strip().replace("_", " ").replace("-", " ")
        return " ".join(part.capitalize() for part in text.split()) or "Other"

    @staticmethod
    def normalize_mpc_material_allow_list(
        allowed_materials: Optional[set[str]],
        material_universe: set[str],
    ) -> Optional[set[str]]:
        """Collapse a complete MPC material allow-list to no active filter."""
        if allowed_materials is None:
            return None

        selected = set(allowed_materials)
        available = set(material_universe)
        if available and available.issubset(selected):
            return None
        return selected

    def populate_material_filters(self) -> None:
        """Populate material filter checkboxes.

        Frame-backed MPC materials take precedence over scene/environment
        materials.  The checkbox list is UI availability state; it must not
        create an active MPC material filter by itself.
        """
        viz = self.visualizer
        try:
            mpc_core = getattr(viz, "mpc_core", None)
            frame_materials = getattr(viz, "_mpc_material_filter_choices", None) or set()
            material_choices: set[str] = set()

            if mpc_core is not None:
                material_choices.update(mpc_core._get_all_environment_materials())
                material_choices.update(mpc_core._normalize_material_filter_labels(frame_materials))
            else:
                material_choices.update(str(material).strip() for material in frame_materials)

            mat_ids = sorted(material for material in material_choices if material)
            if mat_ids:
                viz._mpc_material_filter_choices = set(mat_ids)
                viz._last_material_keys = tuple(mat_ids)

            active_filter = getattr(viz, "mpc_allowed_materials", None)
            if active_filter is None:
                checked = set(mat_ids)
            elif mpc_core is not None:
                checked = mpc_core._normalize_material_filter_labels(active_filter)
            else:
                checked = set(active_filter)
            if active_filter is not None:
                viz.mpc_allowed_materials = self.normalize_mpc_material_allow_list(
                    set(checked),
                    set(mat_ids),
                )
                if viz.mpc_allowed_materials is None:
                    checked = set(mat_ids)

            if hasattr(viz, "ui_manager") and "mpc" in viz.ui_manager.panels:
                viz.ui_manager.panels["mpc"].set_materials(mat_ids, checked=checked)
                logger.info(
                    "Material filter panel populated with %d material choices.",
                    len(mat_ids),
                )
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.warning("Failed to populate material filter panel: %s", e)
