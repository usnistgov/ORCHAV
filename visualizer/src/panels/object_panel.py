"""Scene-object tree panel for visibility, labels, selection, and camera focus.

The panel presents mesh, target, TX, and RX entries in a grouped Qt model. It
delegates appearance, live node editing, and camera focus to visualizer
services/controllers so tree checkboxes remain a UI projection of canonical
scene-entry dictionaries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QTreeView,
    QVBoxLayout,
)

from shared.logging import get_logger

from .base import BasePanel
from .object_management import NodePropertiesDialog

logger = get_logger("orchav.object_panel")


class ObjectManagementPanel(BasePanel):
    """Build and synchronize the Object Management tree panel."""

    class _ObjectTreeFilter(QSortFilterProxyModel):
        """Filter tree rows by object or material text while keeping parent groups."""

        def __init__(self, parent=None):
            """Initialize the case-insensitive search filter."""
            super().__init__(None)  # QSortFilterProxyModel doesn't need parent
            self._search_text = ""

        def set_search_text(self, text: str) -> None:
            """Set search text and invalidate the proxy filter."""
            self._search_text = (text or "").lower()
            self.invalidateFilter()

        def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
            """Accept matching rows and parent groups with matching descendants."""
            model = self.sourceModel()
            index_name = model.index(source_row, 0, source_parent)
            index_material = model.index(source_row, 1, source_parent)

            if not index_name.isValid():
                return False

            if not self._search_text:
                return True

            values = []
            name_data = model.data(index_name, Qt.DisplayRole)
            if isinstance(name_data, str):
                values.append(name_data.lower())
            material_data = model.data(index_material, Qt.DisplayRole)
            if isinstance(material_data, str):
                values.append(material_data.lower())

            for value in values:
                if value and self._search_text in value:
                    return True

            for child_row in range(model.rowCount(index_name)):
                if self.filterAcceptsRow(child_row, index_name):
                    return True

            return False

    class _ObjectTreeView(QTreeView):
        """Tree view that reapplies column sizing after Qt resize events."""

        def __init__(self, resize_callback, parent=None):
            """Store the callback used to maintain compact checkbox columns."""
            super().__init__(parent)
            self._resize_callback = resize_callback

        def resizeEvent(self, event):
            """Forward resize events, then rebalance the tree columns."""
            super().resizeEvent(event)
            if self._resize_callback is not None:
                self._resize_callback()

    def __init__(self, parent_widget):
        """Initialize source/proxy models and cached scene-entry indexes."""
        super().__init__(parent_widget)
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(["Object", "Material ID", "Label", "Highlight"])
        self.model.itemChanged.connect(self._on_item_changed)

        self.proxy_model = self._ObjectTreeFilter()
        self.proxy_model.setSourceModel(self.model)

        self.entry_items: Dict[int, Dict[str, Optional[QStandardItem]]] = {}
        self.entry_items_by_key: Dict[str, Dict[str, Optional[QStandardItem]]] = {}
        self.current_group_by = "Material ID"
        self.current_search_text = ""
        self._mesh_entries: List[Dict[str, Any]] = []
        self._target_entries: List[Dict[str, Any]] = []
        self._tx_entries: List[Dict[str, Any]] = []
        self._rx_entries: List[Dict[str, Any]] = []
        self._suppress_model_signals = False

    def create_panel(self):
        """Create and return the object management panel."""
        group = self.create_group_box("Object Management")

        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        visibility_layout = QHBoxLayout()
        visibility_layout.setSpacing(6)
        self.widgets["scene_cb"] = QCheckBox("Show Scene")
        self.widgets["scene_cb"].setChecked(True)
        visibility_layout.addWidget(self.widgets["scene_cb"])

        self.widgets["building_labels_cb"] = QCheckBox("Show Scene Labels")
        self.widgets["building_labels_cb"].setChecked(False)
        visibility_layout.addWidget(self.widgets["building_labels_cb"])

        visibility_layout.addStretch()
        layout.addLayout(visibility_layout)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        list_header = QLabel("OBJECT LIST")
        list_header.setStyleSheet("font-weight: bold; font-size: 13px; margin: 5px 0px;")
        layout.addWidget(list_header)

        header_layout = QHBoxLayout()
        self.widgets["header_label"] = QLabel("Loaded 0 objects")
        self.widgets["header_label"].setStyleSheet("font-weight: bold; font-size: 12px;")
        header_layout.addWidget(self.widgets["header_label"])

        self.widgets["target_count_label"] = QLabel("")
        self.widgets["target_count_label"].setProperty("role", "secondary")
        self.widgets["target_count_label"].setStyleSheet("font-size: 11px;")
        header_layout.addWidget(self.widgets["target_count_label"])

        header_layout.addStretch()
        layout.addLayout(header_layout)

        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.widgets["object_search_filter"] = QLineEdit()
        self.widgets["object_search_filter"].setPlaceholderText("Type to search objects...")
        self.widgets["object_search_filter"].textChanged.connect(self._on_search_changed)
        search_layout.addWidget(self.widgets["object_search_filter"], 1)
        layout.addLayout(search_layout)

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Group by:"))
        self.widgets["group_by_combo"] = QComboBox()
        self.widgets["group_by_combo"].addItems(["Material ID", "Type", "Alphabetical", "None"])
        self.widgets["group_by_combo"].currentTextChanged.connect(self._on_group_by_changed)
        filter_layout.addWidget(self.widgets["group_by_combo"])

        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # Large scenes default to collapsed groups, so expose explicit tree controls.
        tree_controls = QHBoxLayout()
        expand_all_btn = QPushButton("Expand All")
        expand_all_btn.clicked.connect(lambda: self._expand_all_groups())

        collapse_all_btn = QPushButton("Collapse All")
        collapse_all_btn.clicked.connect(lambda: self._collapse_all_groups())

        tree_controls.addWidget(expand_all_btn)
        tree_controls.addWidget(collapse_all_btn)
        tree_controls.addStretch()
        layout.addLayout(tree_controls)

        tree = self._ObjectTreeView(self._apply_object_column_widths)
        tree.setModel(self.proxy_model)
        tree.setRootIsDecorated(True)
        tree.setAlternatingRowColors(True)
        tree.setUniformRowHeights(True)
        tree.setSelectionMode(QAbstractItemView.SingleSelection)
        tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tree.doubleClicked.connect(self._on_tree_double_clicked)
        tree.setContextMenuPolicy(Qt.CustomContextMenu)
        tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        self.widgets["object_tree"] = tree
        layout.addWidget(tree)
        self._apply_object_column_widths()

        return group

    def update_object_count(self, total_objects, target_count, tx_count=0, rx_count=0):
        """Update object and node-count summary labels."""
        self.widgets["header_label"].setText(f"Loaded {total_objects} objects")
        badge_parts = []
        if target_count > 0:
            badge_parts.append(f"{target_count} targets")
        if tx_count > 0:
            badge_parts.append(f"{tx_count} TX")
        if rx_count > 0:
            badge_parts.append(f"{rx_count} RX")
        self.widgets["target_count_label"].setText(" | ".join(badge_parts))

    def populate_object_list(
        self,
        mesh_entries,
        target_entries,
        tx_entries=None,
        rx_entries=None,
        search_text="",
        group_by="Material ID",
    ):
        """Load scene entries into the tree model and apply search/grouping state."""
        self._mesh_entries = list(mesh_entries)
        self._target_entries = list(target_entries)
        self._tx_entries = list(tx_entries or [])
        self._rx_entries = list(rx_entries or [])
        self.current_group_by = group_by or "Material ID"
        self.current_search_text = search_text or ""
        self._rebuild_model()
        self.proxy_model.set_search_text(self.current_search_text)
        tree = self.widgets.get("object_tree")
        if tree:
            tree.collapseAll()
            tree.expandToDepth(0)
            self._apply_object_column_widths()

    def apply_search_filter(self, text: str) -> None:
        """Apply a tree search and expand matching groups when needed."""
        self.current_search_text = text or ""
        self.proxy_model.set_search_text(self.current_search_text)
        tree = self.widgets.get("object_tree")
        if tree:
            if self.current_search_text:
                tree.expandAll()
            else:
                tree.collapseAll()
                tree.expandToDepth(0)
            self._apply_object_column_widths()

    def apply_grouping(self, group_by: str) -> None:
        """Regroup cached entries by material, type, alphabet, or a flat bucket."""
        self.current_group_by = group_by or "Material ID"
        self._rebuild_model()
        self.proxy_model.set_search_text(self.current_search_text)
        tree = self.widgets.get("object_tree")
        if tree:
            tree.collapseAll()
            tree.expandToDepth(0)
            self._apply_object_column_widths()

    def sync_entry_state(self, entry: Dict[str, Any]) -> None:
        """Sync one canonical scene entry into its visible tree row."""
        items = self.entry_items.get(id(entry))
        if not items:
            items = self.entry_items_by_key.get(self._entry_identity_key(entry))
        if not items:
            # Scene-entry dictionaries can be replaced during material edits.
            entry_name = entry.get("display_name") or entry.get("name", "")
            for entry_id, stored_items in self.entry_items.items():
                vis_item = stored_items.get("visibility")
                if vis_item:
                    stored_entry = vis_item.data(Qt.UserRole)
                    stored_name = None
                    if isinstance(stored_entry, dict):
                        stored_name = stored_entry.get("display_name") or stored_entry.get("name")
                    if stored_name == entry_name:
                        items = stored_items
                        logger.debug(f"Found entry by name '{entry_name}' for material update")
                        break

        if not items:
            logger.debug(
                f"Could not find entry items for '{entry.get('name', 'Unknown')}' - entry may not be in tree view"
            )
            return

        self._suppress_model_signals = True
        try:
            items["visibility"].setCheckState(
                Qt.Checked if entry.get("visible", True) else Qt.Unchecked
            )

            material_item = items.get("material")
            if material_item is not None:
                material_text = self._entry_material_label(entry)
                material_item.setText(material_text)
                logger.debug(
                    f"Updated material item text to '{material_text}' for '{entry.get('name', 'Unknown')}'"
                )

            label_item = items.get("label")
            if label_item is not None:
                # Both meshes and targets can have labels
                label_item.setCheckState(
                    Qt.Checked if self._effective_label_state(entry) else Qt.Unchecked
                )
            highlight_item = items.get("highlight")
            if highlight_item is not None:
                highlight_item.setCheckState(
                    Qt.Checked if entry.get("highlighted", False) else Qt.Unchecked
                )
        finally:
            self._suppress_model_signals = False
            self._update_group_state(items.get("group"))

    def sync_all_entry_states(self) -> None:
        """Bulk-sync every tree checkbox from the canonical entry dicts.

        Call this after an external operation (e.g. material filter) has
        changed entry visibility/highlight without going through the tree.
        Resolves canonical entries from viz.mesh_entries to avoid stale
        copies held by PySide6 QVariant storage.
        """
        self._suppress_model_signals = True
        updated_groups: set = set()
        try:
            for _entry_id, items in self.entry_items.items():
                vis_item = items.get("visibility")
                if vis_item is None:
                    continue
                entry = vis_item.data(Qt.UserRole)
                if not isinstance(entry, dict):
                    continue
                # Resolve canonical entry from mesh_entries
                canonical = self._object_appearance_service().resolve_canonical_entry(entry)

                vis_item.setCheckState(
                    Qt.Checked if canonical.get("visible", True) else Qt.Unchecked
                )
                highlight_item = items.get("highlight")
                if highlight_item is not None and highlight_item.isCheckable():
                    highlight_item.setCheckState(
                        Qt.Checked if canonical.get("highlighted", False) else Qt.Unchecked
                    )
                label_item = items.get("label")
                if label_item is not None and label_item.isCheckable():
                    label_item.setCheckState(
                        Qt.Checked if self._effective_label_state(canonical) else Qt.Unchecked
                    )

                group = items.get("group")
                if group is not None:
                    updated_groups.add(id(group))
        finally:
            self._suppress_model_signals = False

        # Recompute tri-state for each affected group once
        for _entry_id, items in self.entry_items.items():
            group = items.get("group")
            if group is not None and id(group) in updated_groups:
                self._update_group_state(group)
                updated_groups.discard(id(group))

    # Internal helpers -------------------------------------------------
    def _rebuild_model(self) -> None:
        """Rebuild the source model from cached entries and current grouping."""
        self._suppress_model_signals = True
        self.model.clear()
        self.model.setHorizontalHeaderLabels(["Object", "Material ID", "Label", "Highlight"])
        self.entry_items.clear()
        self.entry_items_by_key.clear()

        groups = self._group_entries(
            self._mesh_entries,
            self._target_entries,
            self._tx_entries,
            self._rx_entries,
            self.current_group_by,
        )
        for group_name, entries in groups:
            group_item = QStandardItem(group_name)
            group_item.setEditable(False)
            group_item.setSelectable(True)
            group_item.setCheckable(True)
            group_item.setAutoTristate(True)
            group_item.setData(group_name, Qt.UserRole)

            placeholders = [QStandardItem() for _ in range(3)]
            for placeholder in placeholders:
                placeholder.setFlags(Qt.ItemIsEnabled)
            self.model.appendRow([group_item, *placeholders])
            self._populate_group_children(group_item, entries)
            self._update_group_state(group_item)

        self._suppress_model_signals = False

    def _group_entries(
        self, mesh_entries, target_entries, tx_entries, rx_entries, group_by: str
    ) -> List[Tuple[str, List[Dict[str, Any]]]]:
        """Return sorted tree groups for scene, target, TX, and RX entries."""
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        def add_entry(entry: Dict[str, Any], *, default_type: str = "mesh") -> None:
            """Assign one entry to the current grouping bucket."""
            name = entry.get("display_name") or entry.get("name", "Unnamed")
            if group_by in {"Material", "Material ID"}:
                key = self._entry_material_label(entry)
            elif group_by == "Type":
                key = self._entry_type_label(entry, default_type=default_type)
            elif group_by == "Alphabetical":
                key = (name[:1] or "A").upper()
            else:
                key = "Objects"
            grouped.setdefault(key, []).append(entry)

        for entry in mesh_entries:
            add_entry(entry, default_type="mesh")
        for entry in target_entries:
            add_entry(entry, default_type="target")
        for entry in tx_entries:
            add_entry(entry, default_type="tx")
        for entry in rx_entries:
            add_entry(entry, default_type="rx")

        ordered = []
        for key in sorted(grouped.keys()):
            ordered.append(
                (
                    key,
                    sorted(grouped[key], key=lambda e: e.get("display_name") or e.get("name", "")),
                )
            )
        return ordered

    def _populate_group_children(
        self, group_item: QStandardItem, entries: List[Dict[str, Any]]
    ) -> None:
        """Append tree rows for entries and cache item lookups by identity."""
        for entry in entries:
            display_name = entry.get("display_name") or entry.get("name", "Unnamed")
            name_item = QStandardItem(display_name)
            name_item.setEditable(False)
            name_item.setSelectable(True)
            name_item.setCheckable(True)
            name_item.setData(entry, Qt.UserRole)
            name_item.setCheckState(Qt.Checked if entry.get("visible", True) else Qt.Unchecked)

            material_text = self._entry_material_label(entry)
            material_item = QStandardItem(material_text)
            material_item.setEditable(False)
            material_item.setToolTip(
                "Assigned scene material ID. Use the Materials panel for visual appearance."
            )
            label_item = QStandardItem()
            label_item.setEditable(False)
            if entry.get("supports_label_toggle", True):
                label_item.setCheckable(True)
                label_item.setData(entry, Qt.UserRole)
                label_item.setCheckState(
                    Qt.Checked if self._effective_label_state(entry) else Qt.Unchecked
                )
            else:
                label_item.setFlags(Qt.ItemIsEnabled)

            highlight_item = QStandardItem()
            highlight_item.setEditable(False)
            if entry.get("supports_highlight_toggle", True):
                highlight_item.setCheckable(True)
                highlight_item.setData(entry, Qt.UserRole)
                highlight_item.setCheckState(
                    Qt.Checked if entry.get("highlighted", False) else Qt.Unchecked
                )
            else:
                highlight_item.setFlags(Qt.ItemIsEnabled)
            label_item.setTextAlignment(Qt.AlignCenter)
            highlight_item.setTextAlignment(Qt.AlignCenter)

            group_item.appendRow([name_item, material_item, label_item, highlight_item])
            self.entry_items[id(entry)] = {
                "visibility": name_item,
                "material": material_item,
                "label": label_item,
                "highlight": highlight_item,
                "group": group_item,
            }
            self.entry_items_by_key[self._entry_identity_key(entry)] = self.entry_items[id(entry)]

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
            "transmitter": "Transmitters",
            "rx": "Receivers",
            "receiver": "Receivers",
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

    def _apply_object_column_widths(self) -> None:
        """Balance text columns while keeping label/highlight columns compact."""
        tree = self.widgets.get("object_tree")
        if tree is None:
            return
        header = tree.header()

        def header_width(text: str, minimum: int) -> int:
            """Compute a readable fixed width for compact checkbox headers."""
            return max(minimum, header.fontMetrics().horizontalAdvance(text) + 24)

        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        viewport_width = tree.viewport().width()
        if viewport_width <= 0:
            viewport_width = tree.width()
        label_width = header_width("Label", 54)
        highlight_width = header_width("Highlight", 76)
        object_width = max(96, int(viewport_width * 0.34))
        material_width = max(84, viewport_width - object_width - label_width - highlight_width - 8)
        tree.setColumnWidth(0, object_width)
        tree.setColumnWidth(1, material_width)
        tree.setColumnWidth(2, label_width)
        tree.setColumnWidth(3, highlight_width)

    def _entry_identity_key(self, entry: Dict[str, Any]) -> str:
        """Build a stable fallback key for scene entries whose dict identity changes."""
        if not isinstance(entry, dict):
            return f"obj:{id(entry)}"
        object_key = entry.get("object_key")
        if object_key:
            return f"key:{object_key}"
        object_id = entry.get("object_id")
        if object_id:
            return f"id:{object_id}"
        entry_type = entry.get("entry_type")
        node_index = entry.get("node_index")
        if entry_type in {"tx", "rx"} and node_index is not None:
            return f"node:{entry_type}:{int(node_index)}"
        mesh = entry.get("mesh")
        if mesh is not None:
            return f"mesh:{id(mesh)}"
        name = entry.get("name")
        if name:
            return f"name:{name}"
        return f"obj:{id(entry)}"

    def _object_appearance_service(self) -> Any:
        """Return the service that applies visibility, label, and highlight state."""
        return self.parent.object_appearance_service

    def _effective_label_state(self, entry: Dict[str, Any]) -> bool:
        """Combine per-entry label state with global TX/RX/target label toggles."""
        if not isinstance(entry, dict):
            return False
        if not entry.get("supports_label_toggle", True):
            return False
        entry_enabled = bool(entry.get("show_label", True))
        entry_type = entry.get("entry_type", "mesh")
        app_state = getattr(self.parent, "app_state", None)
        if entry_type == "target":
            global_enabled = bool(getattr(app_state, "show_target_labels", True))
        elif entry_type in {"tx", "rx"}:
            global_enabled = bool(getattr(app_state, "show_labels", True))
        else:
            global_enabled = True
        return bool(global_enabled and entry_enabled)

    def _entry_from_index(self, proxy_index: QModelIndex) -> Optional[Dict[str, Any]]:
        """Resolve a proxy-model index back to its scene-entry dictionary."""
        if not proxy_index.isValid():
            return None
        source_index = self.proxy_model.mapToSource(proxy_index)
        item = self.model.itemFromIndex(source_index)
        if item is None:
            return None
        data = item.data(Qt.UserRole)
        return data if isinstance(data, dict) else None

    def _on_item_changed(self, item: QStandardItem) -> None:
        """Translate tree checkbox changes into object-appearance service calls."""
        if self._suppress_model_signals:
            return
        if not item.isCheckable():
            return

        entry = item.data(Qt.UserRole)
        column = item.column()

        if not isinstance(entry, dict):
            if column == 0:
                self._propagate_group_state(item, item.checkState())
            return

        if column == 0:
            visible = item.checkState() == Qt.Checked
            self._object_appearance_service().set_object_visibility(entry, visible)
        elif column == 2:
            # Both meshes and targets can have labels
            self._object_appearance_service().set_building_label_visibility(
                entry, item.checkState() == Qt.Checked
            )
        elif column == 3:
            self._object_appearance_service().set_object_highlight(
                entry, item.checkState() == Qt.Checked
            )

        self._update_group_state(item.parent())

    def _propagate_group_state(self, group_item: QStandardItem, state: int) -> None:
        """Apply a group checkbox through the mixed-entry appearance service."""
        if group_item is None:
            return
        visible = state == Qt.Checked
        entries: list[Dict[str, Any]] = []
        self._suppress_model_signals = True
        try:
            for row in range(group_item.rowCount()):
                child = group_item.child(row, 0)
                if child and child.isCheckable():
                    child.setCheckState(state)
                    entry = child.data(Qt.UserRole)
                    if isinstance(entry, dict):
                        entries.append(entry)
        finally:
            self._suppress_model_signals = False

        if entries:
            self._object_appearance_service().set_object_visibility_batch(entries, visible)

        self._update_group_state(group_item)

    def _update_group_state(self, group_item: Optional[QStandardItem]) -> None:
        """Refresh parent tri-state checkboxes from child visibility state."""
        if group_item is None:
            return
        total = 0
        checked = 0
        for row in range(group_item.rowCount()):
            child = group_item.child(row, 0)
            if child and child.isCheckable():
                total += 1
                if child.checkState() == Qt.Checked:
                    checked += 1
        self._suppress_model_signals = True
        try:
            if total == 0 or checked == 0:
                group_item.setCheckState(Qt.Unchecked)
            elif checked == total:
                group_item.setCheckState(Qt.Checked)
            else:
                group_item.setCheckState(Qt.PartiallyChecked)
        finally:
            self._suppress_model_signals = False
        parent_item = group_item.parent()
        if parent_item:
            self._update_group_state(parent_item)

    def _on_search_changed(self, text: str) -> None:
        """Forward search-text edits to the tree filter."""
        self.apply_search_filter(text)

    def _on_group_by_changed(self, text: str) -> None:
        """Forward group-mode changes to the tree model."""
        self.apply_grouping(text)

    def _on_tree_double_clicked(self, proxy_index: QModelIndex) -> None:
        """Focus the camera on a double-clicked scene entry."""
        entry = self._entry_from_index(proxy_index)
        if entry:
            self._focus_camera_on_entry(entry)

    def _on_tree_context_menu(self, pos) -> None:
        """Build navigation and supported live-node actions for one entry."""
        tree = self.widgets.get("object_tree")
        if tree is None:
            return
        proxy_index = tree.indexAt(pos)
        entry = self._entry_from_index(proxy_index)
        if not entry:
            return

        menu = QMenu(tree)
        entry_type = entry.get("entry_type")
        is_online = self._is_online_mode()
        properties_action = None
        if entry_type in ["target", "tx", "rx"]:
            properties_action = menu.addAction("Node Properties…")
        if properties_action is not None and not is_online:
            menu.addSeparator()
            menu.addAction("Connect to generator for editing").setEnabled(False)

        # Focus camera on object (always available)
        focus_camera_action = None
        menu.addSeparator()
        focus_camera_action = menu.addAction("Focus Camera")

        selected = menu.exec(tree.viewport().mapToGlobal(pos))
        if selected == properties_action:
            self._show_node_properties_dialog(entry)
        elif selected == focus_camera_action:
            self._focus_camera_on_entry(entry)

    def _focus_camera_on_entry(self, entry: Dict[str, Any]) -> None:
        """Focus the camera on the selected object by adjusting the lookat target."""
        import numpy as np

        renderer = getattr(self.parent, "renderer", None)
        if renderer is None or not hasattr(renderer, "get_camera_state"):
            return

        center = None
        for key in ("current_center", "position", "original_center"):
            val = entry.get(key)
            if val is not None:
                center = np.asarray(val, dtype=float)
                break

        if center is None:
            return

        cam = renderer.get_camera_state()
        if cam is None:
            return
        from ..types.camera_state import CameraState

        if not isinstance(cam, CameraState):
            return
        renderer.set_camera_state(
            CameraState(
                eye=cam.eye, lookat=tuple(float(x) for x in center), up=cam.up, fov_deg=cam.fov_deg
            )
        )

    def _is_online_mode(self) -> bool:
        """Return whether the current frame source supports live gRPC edits."""
        try:
            from ..io.frame_sources import LiveGrpcSource

            return hasattr(self.parent, "frame_source") and isinstance(
                self.parent.frame_source, LiveGrpcSource
            )
        except ImportError:
            return False

    def _show_node_properties_dialog(self, entry: Dict[str, Any]) -> None:
        """Show supported live properties for a TX, RX, or target entry."""
        entry_type = entry.get("entry_type")
        if entry_type in ("target", "tx", "rx"):
            if not self._is_online_mode():
                QMessageBox.information(
                    self.widgets.get("object_tree"),
                    "Live gRPC Mode Required",
                    "Editing target/TX/RX properties requires an active live gRPC connection.",
                )
                return
            dialog = NodePropertiesDialog(self.parent, entry)
            if dialog.exec() == QDialog.Accepted:
                values = dialog.get_result()
                self.parent.scene_edit_service.edit_node_properties(entry, values)
            return

        QMessageBox.information(
            self.widgets.get("object_tree"), "Properties", "Unsupported entry type."
        )

    def _expand_all_groups(self) -> None:
        """Expand all group items in the tree."""
        tree = self.widgets.get("object_tree")
        if tree is None:
            return
        tree.expandAll()

    def _collapse_all_groups(self) -> None:
        """Collapse all group items in the tree."""
        tree = self.widgets.get("object_tree")
        if tree is None:
            return
        tree.collapseAll()
