"""Spacious, model-neutral Qt shell for inspecting MPC paths.

The window deliberately owns no canonical MPC data and performs no indexing.
It exposes user intent through signals and accepts the shared Explorer model
only while an active session has a successfully presented frame.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from PySide6.QtCore import QEvent, QModelIndex, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStyledItemDelegate,
    QTableView,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

SCOPE_OPTIONS = (
    ("All paths", "all"),
    ("Filtered paths", "filtered"),
    ("Actually rendered paths", "rendered"),
)

GROUPING_OPTIONS = (
    ("None (global order)", "none"),
    ("TX -> RX", "tx_rx"),
    ("RX -> TX", "rx_tx"),
    ("Interactions", "interactions"),
    ("Interaction mix", "interaction_mix"),
    ("First material", "first_material"),
    ("Delay band", "delay_band"),
    ("Path-loss band", "path_loss_band"),
)

GROUPING_SORT_FIELDS: Mapping[str, tuple[str, ...]] = {
    "none": (),
    "tx_rx": ("tx", "rx"),
    "rx_tx": ("rx", "tx"),
    "interactions": ("interactions",),
    "interaction_mix": ("interaction_mix",),
    "first_material": ("first_material",),
    "delay_band": ("delay_band",),
    "path_loss_band": ("path_loss_band",),
}

SORT_PRESETS: Mapping[str, tuple[str, tuple[tuple[str, bool], ...]]] = {
    "tx_rx_strongest": (
        "Per TX/RX — strongest first",
        (
            ("tx", True),
            ("rx", True),
            ("path_loss", True),
            ("delay", True),
        ),
    ),
    "tx_rx_earliest": (
        "Per TX/RX — earliest first",
        (
            ("tx", True),
            ("rx", True),
            ("delay", True),
            ("path_loss", True),
        ),
    ),
    "strongest_overall": (
        "Global — strongest first",
        (("path_loss", True), ("delay", True)),
    ),
    "earliest_overall": (
        "Global — earliest first",
        (("delay", True), ("path_loss", True)),
    ),
    "interactions_strongest": (
        "By interaction count — strongest first",
        (("interactions", True), ("path_loss", True)),
    ),
    "interaction_mix_strongest": (
        "By interaction mix — strongest first",
        (("interaction_mix", True), ("path_loss", True)),
    ),
    "first_material_strongest": (
        "By first material — strongest first",
        (("first_material", True), ("path_loss", True)),
    ),
    "delay_band_strongest": (
        "By delay band — strongest first",
        (("delay_band", True), ("path_loss", True)),
    ),
    "loss_band_earliest": (
        "By path-loss band — earliest first",
        (("path_loss_band", True), ("delay", True)),
    ),
}

SORT_PRESET_DETAILS: Mapping[str, str] = {
    "tx_rx_strongest": ("Separate each TX/RX pair, then show its lowest-loss paths first."),
    "tx_rx_earliest": ("Separate each TX/RX pair, then show its lowest-delay paths first."),
    "strongest_overall": "One global list ordered by path loss, lowest first.",
    "earliest_overall": "One global list ordered by delay, lowest first.",
    "interactions_strongest": (
        "Separate paths by bounce count, then order each group by path loss."
    ),
    "interaction_mix_strongest": (
        "Separate paths by mechanism mix, then order each group by path loss."
    ),
    "first_material_strongest": (
        "Separate paths by first bounce material, then order each group by path loss."
    ),
    "delay_band_strongest": (
        "Separate paths into 10 ns delay bands, then show the strongest in each band."
    ),
    "loss_band_earliest": (
        "Separate paths into 10 dB path-loss bands, then show the earliest in each band."
    ),
}

PRESET_GROUPINGS: Mapping[str, str] = {
    "tx_rx_strongest": "tx_rx",
    "tx_rx_earliest": "tx_rx",
    "strongest_overall": "none",
    "earliest_overall": "none",
    "interactions_strongest": "interactions",
    "interaction_mix_strongest": "interaction_mix",
    "first_material_strongest": "first_material",
    "delay_band_strongest": "delay_band",
    "loss_band_earliest": "path_loss_band",
}

SORT_COLUMNS = (
    ("TX", "tx"),
    ("RX", "rx"),
    ("Path loss", "path_loss"),
    ("Delay", "delay"),
    ("Interactions", "interactions"),
    ("Interaction mix", "interaction_mix"),
    ("First material", "first_material"),
    ("Delay band", "delay_band"),
    ("Path-loss band", "path_loss_band"),
    ("AoD azimuth (world deg)", "aod_azimuth"),
    ("AoD elevation (world deg)", "aod_elevation"),
    ("AoA azimuth (world deg)", "aoa_azimuth"),
    ("AoA elevation (world deg)", "aoa_elevation"),
    ("Geometric length", "geometric_length"),
    ("Stretch ratio", "stretch_ratio"),
    ("Excess delay", "excess_delay"),
    ("Strength rank", "strength_rank"),
    ("Loss delta from strongest", "relative_path_loss"),
    ("Relative power proxy", "relative_power"),
)

SORT_LABELS = {key: label for label, key in SORT_COLUMNS}


class _GroupBoundaryDelegate(QStyledItemDelegate):
    """Draw one inexpensive top rule where a flat-table group changes."""

    def __init__(self, boundary_role: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._boundary_role = int(boundary_role)

    def paint(self, painter: Any, option: Any, index: QModelIndex) -> None:
        super().paint(painter, option, index)
        if index.row() <= 0 or not bool(index.data(self._boundary_role)):
            return
        painter.save()
        pen = QPen(option.palette.mid().color())
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(option.rect.topLeft(), option.rect.topRight())
        painter.restore()


class MpcExplorerWindow(QMainWindow):
    """Resizable MPC Explorer shell backed by one shared table model."""

    activityChanged = Signal(bool)
    closing = Signal()
    scopeChanged = Signal(str)
    groupingChanged = Signal(str)
    sortPresetChanged = Signal(str)
    sortClausesChanged = Signal(object)
    filtersChanged = Signal(object)
    columnVisibilityChanged = Signal(str, bool)
    pathSelectionRequested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build controls without constructing a catalog, model, or worker."""
        super().__init__(parent)
        self.setWindowTitle("MPC Explorer")
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setMinimumSize(820, 520)
        self.resize(1180, 760)

        self._activity_state = False
        self._selection_model: Any = None
        self._column_menu = QMenu(self)
        self._custom_sort_clauses: list[tuple[str, bool]] = []
        self._applied_custom_sort_clauses: tuple[tuple[str, bool], ...] = ()
        self._applied_custom_grouping = "tx_rx"
        self._has_applied_custom_order = False
        self._custom_draft_dirty = False
        self._optional_start_column: int | None = None
        self._group_boundary_delegate: Any = None
        self._programmatic_path_selection = False
        self._material_filter_hint = "Material IDs are listed after a frame is indexed."
        self._visible_optional_columns: set[str] = set()
        self._displayed_sort_clauses: tuple[tuple[str, bool], ...] = ()
        self._displayed_sort_removable = False

        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.addLayout(self._create_primary_controls())
        root.addWidget(self._create_filter_drawer())
        root.addWidget(self._create_ordering_tabs())
        root.addWidget(self._create_sort_chip_row())
        root.addWidget(self._create_content_splitter(), 1)
        root.addLayout(self._create_footer())
        self.setCentralWidget(central)

        self._seed_custom_from_preset("tx_rx_strongest", mark_applied=True)
        self._set_preset_chips("tx_rx_strongest")
        self._update_preset_description()
        self._update_custom_sort_enabled()
        self.set_status("Waiting for a successfully presented MPC frame...")

    def _create_primary_controls(self) -> QHBoxLayout:
        """Create controls shared by preset and custom ordering modes."""
        row = QHBoxLayout()
        row.setSpacing(6)

        row.addWidget(QLabel("Scope:"))
        self.scope_combo = QComboBox()
        for label, key in SCOPE_OPTIONS:
            self.scope_combo.addItem(label, key)
        self.scope_combo.currentIndexChanged.connect(self._emit_scope)
        row.addWidget(self.scope_combo)
        row.addStretch(1)

        self.column_button = QToolButton()
        self.column_button.setText("Columns")
        self.column_button.setPopupMode(QToolButton.InstantPopup)
        self.column_button.setMenu(self._column_menu)
        row.addWidget(self.column_button)

        self.filter_button = QToolButton()
        self.filter_button.setText("Filters")
        self.filter_button.setCheckable(True)
        self.filter_button.toggled.connect(self._set_filter_drawer_visible)
        row.addWidget(self.filter_button)
        return row

    def _create_ordering_tabs(self) -> QTabWidget:
        """Separate complete preset recipes from explicitly authored orders."""
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        self.ordering_tabs = tabs

        preset_page = QWidget()
        preset_layout = QVBoxLayout(preset_page)
        preset_layout.setContentsMargins(6, 4, 6, 4)
        preset_layout.setSpacing(3)
        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        preset_row.addWidget(QLabel("Preset recipe:"))
        self.preset_combo = QComboBox()
        for key, (label, _clauses) in SORT_PRESETS.items():
            self.preset_combo.addItem(label, key)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self.preset_combo, 1)
        preset_layout.addLayout(preset_row)
        self.preset_description_label = QLabel()
        self.preset_description_label.setWordWrap(True)
        preset_layout.addWidget(self.preset_description_label)
        tabs.addTab(preset_page, "Preset recipes")

        custom_page = QWidget()
        custom_layout = QVBoxLayout(custom_page)
        custom_layout.setContentsMargins(6, 4, 6, 4)
        custom_layout.setSpacing(3)
        custom_layout.addLayout(self._create_sort_controls())
        self.custom_sort_help_label = QLabel(
            "Group controls the locked first level. The first sort clause is primary; "
            "later clauses only break ties. Path ID is the final implicit tie-breaker."
        )
        self.custom_sort_help_label.setWordWrap(True)
        custom_layout.addWidget(self.custom_sort_help_label)
        tabs.addTab(custom_page, "Custom order")

        tabs.currentChanged.connect(self._on_ordering_mode_changed)
        return tabs

    def _create_filter_drawer(self) -> QWidget:
        """Create unrestricted text filters without artificial numeric bounds."""
        drawer = QFrame()
        drawer.setFrameShape(QFrame.StyledPanel)
        drawer.setVisible(False)
        self.filter_drawer = drawer
        layout = QGridLayout(drawer)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(4)

        self.filter_edits: dict[str, QLineEdit] = {}
        fields = (
            ("TX IDs", "tx_ids", "e.g. 0, 2"),
            ("RX IDs", "rx_ids", "e.g. 1, 3"),
            ("Loss min (dB)", "path_loss_min_db", "unbounded"),
            ("Loss max (dB)", "path_loss_max_db", "unbounded"),
            ("Delay min (ns)", "delay_min_ns", "unbounded"),
            ("Delay max (ns)", "delay_max_ns", "unbounded"),
            ("Interactions min", "interaction_count_min", "unbounded"),
            ("Interactions max", "interaction_count_max", "unbounded"),
            ("Contains mechanisms", "contains_interactions", "e.g. 1, 2"),
            ("Pure mechanism", "pure_interaction", "single ID"),
            ("Exact sequence", "exact_interaction_sequence", "e.g. 1, 2, 1"),
            ("First materials", "first_material_ids", "e.g. 3, 7"),
        )
        for index, (label, key, placeholder) in enumerate(fields):
            row = index // 4
            pair = index % 4
            column = pair * 2
            layout.addWidget(QLabel(f"{label}:"), row, column)
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            edit.returnPressed.connect(self._apply_filters)
            self.filter_edits[key] = edit
            layout.addWidget(edit, row, column + 1)

        button_row = (len(fields) + 3) // 4
        self.mixed_only_checkbox = QCheckBox("Mixed mechanisms only")
        layout.addWidget(self.mixed_only_checkbox, button_row, 0, 1, 2)
        self.filter_value_hint = QLabel(
            "Mechanism IDs: 0 LoS, 1 Specular, 2 Diffuse, 4 Refraction, "
            "8 Diffraction, 99 Virtual. " + self._material_filter_hint
        )
        self.filter_value_hint.setWordWrap(True)
        layout.addWidget(self.filter_value_hint, button_row + 1, 0, 1, 8)
        layout.setColumnStretch(6, 1)
        layout.setColumnStretch(7, 2)

        self.apply_filters_button = QPushButton("Apply filters")
        self.apply_filters_button.clicked.connect(self._apply_filters)
        layout.addWidget(self.apply_filters_button, button_row, 6)
        self.clear_filters_button = QPushButton("Clear filters")
        self.clear_filters_button.clicked.connect(self._clear_filters)
        layout.addWidget(self.clear_filters_button, button_row, 7)
        return drawer

    def _create_sort_controls(self) -> QVBoxLayout:
        """Create the compact custom compound-sort editor."""
        layout = QVBoxLayout()
        layout.setSpacing(3)

        group_row = QHBoxLayout()
        group_row.setSpacing(6)
        group_row.addWidget(QLabel("Group rows by:"))
        self.grouping_combo = QComboBox()
        for label, key in GROUPING_OPTIONS:
            self.grouping_combo.addItem(label, key)
        self.grouping_combo.setCurrentIndex(1)
        self.grouping_combo.setToolTip(
            "The group is a locked primary classification. Choose None for one " "global order."
        )
        self.grouping_combo.currentIndexChanged.connect(self._on_custom_grouping_changed)
        group_row.addWidget(self.grouping_combo)
        group_row.addStretch(1)
        layout.addLayout(group_row)

        clause_row = QHBoxLayout()
        clause_row.setSpacing(6)
        clause_row.addWidget(QLabel("Sort field:"))

        self.sort_column_combo = QComboBox()
        for label, key in SORT_COLUMNS:
            self.sort_column_combo.addItem(label, key)
        clause_row.addWidget(self.sort_column_combo, 1)

        self.sort_direction_combo = QComboBox()
        self.sort_direction_combo.addItem("Ascending", True)
        self.sort_direction_combo.addItem("Descending", False)
        clause_row.addWidget(self.sort_direction_combo)

        self.set_primary_sort_button = QPushButton("Use as primary")
        self.set_primary_sort_button.setToolTip("Move this field to the front of the custom order.")
        self.set_primary_sort_button.clicked.connect(self._set_custom_sort_primary)
        clause_row.addWidget(self.set_primary_sort_button)

        self.add_sort_clause_button = QPushButton("Add clause")
        self.add_sort_clause_button.setToolTip("Append this field as a lower-priority tie-breaker.")
        self.add_sort_clause_button.clicked.connect(self._add_sort_clause)
        clause_row.addWidget(self.add_sort_clause_button)

        self.apply_custom_sort_button = QPushButton("Apply custom")
        self.apply_custom_sort_button.setEnabled(False)
        self.apply_custom_sort_button.setToolTip("Custom sorts require two to four clauses.")
        self.apply_custom_sort_button.clicked.connect(self._apply_custom_sort)
        clause_row.addWidget(self.apply_custom_sort_button)

        self.clear_custom_sort_button = QPushButton("Reset custom")
        self.clear_custom_sort_button.setToolTip("Reset to one global strongest-first order.")
        self.clear_custom_sort_button.clicked.connect(self._clear_custom_sort)
        clause_row.addWidget(self.clear_custom_sort_button)
        layout.addLayout(clause_row)
        return layout

    def _create_sort_chip_row(self) -> QWidget:
        """Create the visible compound-order chip strip."""
        container = QFrame()
        container.setFrameShape(QFrame.StyledPanel)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(4)
        self.sort_chip_caption = QLabel("Effective order:")
        layout.addWidget(self.sort_chip_caption)
        self.sort_chip_layout = layout
        self._sort_chip_stretch = QWidget()
        self._sort_chip_stretch.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(self._sort_chip_stretch)
        return container

    def _create_content_splitter(self) -> QSplitter:
        """Create the read-only path table and selected-path details pane."""
        splitter = QSplitter(Qt.Vertical)

        self.table = QTableView()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.verticalHeader().setMinimumSectionSize(18)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        splitter.addWidget(self.table)

        details_container = QWidget()
        details_layout = QVBoxLayout(details_container)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(3)
        details_layout.addWidget(QLabel("Selected path details"))
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumBlockCount(500)
        self.details.setPlaceholderText("Select a path in the table or pygfx viewport.")
        details_layout.addWidget(self.details)
        splitter.addWidget(details_container)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        return splitter

    def _create_footer(self) -> QHBoxLayout:
        """Create the result count and background-work status row."""
        row = QHBoxLayout()
        self.count_label = QLabel("0 / 0 paths")
        row.addWidget(self.count_label)
        row.addStretch()
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.status_label, 1)
        return row

    def set_model(
        self,
        model: Any | None,
        *,
        optional_start_column: int | None = None,
    ) -> None:
        """Install or release the shared Explorer model."""
        if self.table.model() is model:
            return
        if self._selection_model is not None:
            try:
                self._selection_model.currentChanged.disconnect(self._on_current_changed)
            except (RuntimeError, TypeError):
                pass
        if self._group_boundary_delegate is not None:
            self._group_boundary_delegate.deleteLater()
            self._group_boundary_delegate = None
        self._selection_model = None
        self._optional_start_column = optional_start_column
        self.table.setSortingEnabled(False)
        self.table.setModel(model)
        if model is None:
            self._rebuild_column_menu()
            return
        self._selection_model = self.table.selectionModel()
        if self._selection_model is not None:
            self._selection_model.currentChanged.connect(self._on_current_changed)
        if optional_start_column is not None:
            for column in range(max(0, int(optional_start_column)), model.columnCount()):
                columns = getattr(model, "columns", ())
                try:
                    column_key = str(getattr(columns[column], "value", columns[column]))
                except (IndexError, TypeError):
                    column_key = ""
                self.table.setColumnHidden(
                    column,
                    column_key not in self._visible_optional_columns,
                )
        boundary_role = getattr(model, "GroupBoundaryRole", None)
        if boundary_role is not None:
            self._group_boundary_delegate = _GroupBoundaryDelegate(
                int(boundary_role),
                self.table,
            )
            self.table.setItemDelegate(self._group_boundary_delegate)
        self._rebuild_column_menu()
        self.table.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        self.table.setSortingEnabled(True)

    def release_model(self) -> Any | None:
        """Detach and return the current model without deleting it."""
        model = self.table.model()
        self.set_model(None)
        return model

    def set_counts(self, visible: int, total: int) -> None:
        """Show current query rows versus the canonical frame population."""
        self.count_label.setText(f"{max(0, int(visible)):,} / {max(0, int(total)):,} paths")

    def set_status(self, text: str) -> None:
        """Update the compact lifecycle/query status message."""
        self.status_label.setText(str(text))

    def set_material_filter_options(self, options: Mapping[int, str]) -> None:
        """Show frame-local material IDs without constructing per-path strings."""
        entries = []
        for material_id, name in sorted(options.items()):
            label = str(name or "").strip() or f"Material {int(material_id)}"
            entries.append(f"{int(material_id)} {label}")
        if entries:
            material_hint = "Materials: " + ", ".join(entries)
        else:
            material_hint = "No named bounce materials are available in this frame."
        self._material_filter_hint = material_hint
        self.filter_value_hint.setText(
            "Mechanism IDs: 0 LoS, 1 Specular, 2 Diffuse, 4 Refraction, "
            "8 Diffraction, 99 Virtual. " + material_hint
        )

    def set_details(self, details: str | Mapping[str, Any] | Iterable[tuple[Any, Any]]) -> None:
        """Display selected-path information without owning its computation."""
        if isinstance(details, str):
            text = details
        else:
            items = details.items() if isinstance(details, Mapping) else details
            text = "\n".join(f"{key}: {value}" for key, value in items)
        self.details.setPlainText(text)

    def clear_details(self) -> None:
        """Clear details when selection or presented-frame identity changes."""
        self.details.clear()

    def current_scope(self) -> str:
        """Return the normalized active scope key."""
        return str(self.scope_combo.currentData() or "all")

    def current_grouping(self) -> str:
        """Return the normalized active grouping key."""
        return str(self.grouping_combo.currentData() or "none")

    def current_preset(self) -> str:
        """Return the normalized active sort-preset key."""
        return str(self.preset_combo.currentData() or "tx_rx_strongest")

    def current_ordering_mode(self) -> str:
        """Return whether a complete preset or a custom order is active."""
        return "custom" if self.ordering_tabs.currentIndex() == 1 else "preset"

    def refresh_sort_context(self) -> None:
        """Re-render chips after a programmatic grouping change."""
        self._render_sort_chips()

    def select_path(self, path_id: int) -> bool:
        """Select and reveal a canonical path when the model can resolve it."""
        model = self.table.model()
        if model is None:
            return False
        row = self._model_row_for_path(model, int(path_id))
        if row is None or row < 0:
            return False
        ensure_fetched = getattr(model, "ensure_row_fetched", None)
        if not callable(ensure_fetched):
            ensure_fetched = getattr(model, "ensure_row_loaded", None)
        if callable(ensure_fetched):
            ensure_fetched(row)
        if row >= model.rowCount():
            return False
        index = model.index(row, 0)
        if not index.isValid():
            return False
        self._programmatic_path_selection = True
        try:
            self.table.setCurrentIndex(index)
            self.table.selectRow(row)
            self.table.scrollTo(index, QAbstractItemView.PositionAtCenter)
        finally:
            self._programmatic_path_selection = False
        return True

    def _emit_scope(self) -> None:
        self.scopeChanged.emit(self.current_scope())

    def _on_preset_changed(self) -> None:
        preset = self.current_preset()
        self._update_preset_description()
        if self.current_ordering_mode() != "preset":
            return
        self._set_preset_chips(preset)
        self.sortPresetChanged.emit(preset)

    def _on_ordering_mode_changed(self, index: int) -> None:
        """Activate exactly one ordering model when the user changes tabs."""
        if int(index) == 0:
            self._set_preset_chips(self.current_preset())
            self.sortPresetChanged.emit(self.current_preset())
        else:
            if not self._has_applied_custom_order:
                self._seed_custom_from_preset(
                    self.current_preset(),
                    mark_applied=True,
                )
            self._restore_applied_custom_order()
            self.groupingChanged.emit(self.current_grouping())
            self.sortClausesChanged.emit(tuple(self._custom_sort_clauses))
        self._update_custom_sort_enabled()

    def _on_custom_grouping_changed(self) -> None:
        """Treat grouping edits as a custom draft until Apply is pressed."""
        if self.current_ordering_mode() != "custom":
            return
        grouping_fields = set(GROUPING_SORT_FIELDS.get(self.current_grouping(), ()))
        self._custom_sort_clauses = [
            clause for clause in self._custom_sort_clauses if clause[0] not in grouping_fields
        ]
        self._fill_custom_sort_clauses()
        self._mark_custom_sort_dirty()

    def _set_custom_sort_primary(self) -> None:
        """Insert or move the selected field to the first custom clause."""
        key = str(self.sort_column_combo.currentData())
        ascending = bool(self.sort_direction_combo.currentData())
        if key in GROUPING_SORT_FIELDS.get(self.current_grouping(), ()):
            self.set_status(f"{SORT_LABELS.get(key, key)} is already supplied by the active group.")
            return
        remaining = [clause for clause in self._custom_sort_clauses if clause[0] != key]
        self._custom_sort_clauses = [(key, ascending), *remaining][:4]
        self._fill_custom_sort_clauses()
        self._mark_custom_sort_dirty()

    def _add_sort_clause(self) -> None:
        if len(self._custom_sort_clauses) >= 4:
            return
        key = str(self.sort_column_combo.currentData())
        ascending = bool(self.sort_direction_combo.currentData())
        if key in GROUPING_SORT_FIELDS.get(self.current_grouping(), ()):
            self.set_status(f"{SORT_LABELS.get(key, key)} is already supplied by the active group.")
            return
        if any(existing_key == key for existing_key, _ascending in self._custom_sort_clauses):
            self.set_status(f"{SORT_LABELS.get(key, key)} is already in the custom sort.")
            return
        self._custom_sort_clauses.append((key, ascending))
        self._mark_custom_sort_dirty()

    def _remove_sort_clause(self, index: int) -> None:
        if len(self._custom_sort_clauses) <= 2:
            self.set_status("Keep at least two custom clauses; later clauses only break ties.")
            return
        if 0 <= index < len(self._custom_sort_clauses):
            self._custom_sort_clauses.pop(index)
        self._mark_custom_sort_dirty()

    def _clear_custom_sort(self) -> None:
        with QSignalBlocker(self.grouping_combo):
            none_index = self.grouping_combo.findData("none")
            if none_index >= 0:
                self.grouping_combo.setCurrentIndex(none_index)
        self._custom_sort_clauses = [("path_loss", True), ("delay", True)]
        self._mark_custom_sort_dirty()

    def _apply_custom_sort(self) -> None:
        if 2 <= len(self._custom_sort_clauses) <= 4:
            self._applied_custom_grouping = self.current_grouping()
            self._applied_custom_sort_clauses = tuple(self._custom_sort_clauses)
            self._has_applied_custom_order = True
            self._custom_draft_dirty = False
            self._set_sort_chips(self._custom_sort_clauses, removable=True)
            self._update_custom_sort_enabled()
            self.groupingChanged.emit(self._applied_custom_grouping)
            self.sortClausesChanged.emit(tuple(self._custom_sort_clauses))

    def set_sort_spec(self, sort_spec: Any, *, global_order: bool = False) -> None:
        """Show a worker sort, entering Custom mode for a table-header request."""
        clauses = getattr(sort_spec, "clauses", ())
        normalized = []
        for clause in clauses:
            field = getattr(getattr(clause, "field", None), "value", None)
            direction = getattr(getattr(clause, "direction", None), "value", None)
            if field is None:
                continue
            normalized.append((str(field), str(direction) != "descending"))
        if normalized:
            self._custom_sort_clauses = normalized[:4]
            if global_order:
                with QSignalBlocker(self.grouping_combo):
                    none_index = self.grouping_combo.findData("none")
                    if none_index >= 0:
                        self.grouping_combo.setCurrentIndex(none_index)
            with QSignalBlocker(self.ordering_tabs):
                self.ordering_tabs.setCurrentIndex(1)
            self._applied_custom_grouping = self.current_grouping()
            self._applied_custom_sort_clauses = tuple(self._custom_sort_clauses)
            self._has_applied_custom_order = True
            self._custom_draft_dirty = False
            self._set_sort_chips(self._custom_sort_clauses, removable=True)
            self._update_custom_sort_enabled()

    def _update_custom_sort_enabled(self) -> None:
        valid = 2 <= len(self._custom_sort_clauses) <= 4
        custom_active = self.current_ordering_mode() == "custom"
        self.apply_custom_sort_button.setEnabled(
            valid and custom_active and self._custom_draft_dirty
        )
        self.set_primary_sort_button.setEnabled(custom_active)
        self.add_sort_clause_button.setEnabled(custom_active and len(self._custom_sort_clauses) < 4)
        self.clear_custom_sort_button.setEnabled(custom_active)

    def _update_preset_description(self) -> None:
        preset = self.current_preset()
        self.preset_description_label.setText(SORT_PRESET_DETAILS.get(preset, ""))

    def _seed_custom_from_preset(self, preset: str, *, mark_applied: bool) -> None:
        grouping = PRESET_GROUPINGS.get(preset, "none")
        with QSignalBlocker(self.grouping_combo):
            grouping_index = self.grouping_combo.findData(grouping)
            if grouping_index >= 0:
                self.grouping_combo.setCurrentIndex(grouping_index)
        _label, clauses = SORT_PRESETS.get(preset, SORT_PRESETS["tx_rx_strongest"])
        grouping_fields = set(GROUPING_SORT_FIELDS.get(grouping, ()))
        self._custom_sort_clauses = [
            clause for clause in clauses if clause[0] not in grouping_fields
        ]
        self._fill_custom_sort_clauses()
        if mark_applied:
            self._applied_custom_grouping = grouping
            self._applied_custom_sort_clauses = tuple(self._custom_sort_clauses)
            self._custom_draft_dirty = False

    def _restore_applied_custom_order(self) -> None:
        with QSignalBlocker(self.grouping_combo):
            grouping_index = self.grouping_combo.findData(self._applied_custom_grouping)
            if grouping_index >= 0:
                self.grouping_combo.setCurrentIndex(grouping_index)
        self._custom_sort_clauses = list(self._applied_custom_sort_clauses)
        self._fill_custom_sort_clauses()
        self._custom_draft_dirty = False
        self._set_sort_chips(self._custom_sort_clauses, removable=True)

    def _fill_custom_sort_clauses(self) -> None:
        """Keep a valid two-clause draft without duplicating group fields."""
        grouping_fields = set(GROUPING_SORT_FIELDS.get(self.current_grouping(), ()))
        existing = {key for key, _ascending in self._custom_sort_clauses}
        for key in ("path_loss", "delay", "interactions", "tx", "rx"):
            if len(self._custom_sort_clauses) >= 2:
                break
            if key not in existing and key not in grouping_fields:
                self._custom_sort_clauses.append((key, True))
                existing.add(key)

    def _mark_custom_sort_dirty(self) -> None:
        self._custom_draft_dirty = True
        self._set_sort_chips(self._custom_sort_clauses, removable=True)
        self._update_custom_sort_enabled()

    def _set_preset_chips(self, preset: str) -> None:
        _label, clauses = SORT_PRESETS.get(preset, SORT_PRESETS["tx_rx_strongest"])
        self._set_sort_chips(clauses, removable=False)

    def _set_sort_chips(
        self,
        clauses: Iterable[tuple[str, bool]],
        *,
        removable: bool,
    ) -> None:
        self._displayed_sort_clauses = tuple(clauses)
        self._displayed_sort_removable = bool(removable)
        self._render_sort_chips()

    def _render_sort_chips(self) -> None:
        """Show the actual group-first order rather than only requested clauses."""
        while self.sort_chip_layout.count() > 2:
            item = self.sort_chip_layout.takeAt(1)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        normalized_clauses = self._displayed_sort_clauses
        grouping_fields = GROUPING_SORT_FIELDS.get(self.current_grouping(), ())
        requested = {key: ascending for key, ascending in normalized_clauses}
        effective_clauses = [(key, requested.get(key, True), True, None) for key in grouping_fields]
        effective_clauses.extend(
            (key, ascending, False, index)
            for index, (key, ascending) in enumerate(normalized_clauses)
            if key not in grouping_fields
        )
        custom_mode = self.current_ordering_mode() == "custom"
        if custom_mode and self._custom_draft_dirty:
            caption = "Custom draft — Apply to use:"
        elif custom_mode:
            caption = "Custom effective order:"
        else:
            caption = "Preset effective order:"
        self.sort_chip_caption.setText(caption)
        self.sort_chip_caption.setToolTip(
            "Group fields are locked first. Choose None in Custom order for a " "global sort."
            if grouping_fields
            else "This is one global order."
        )
        can_remove = self._displayed_sort_removable and custom_mode and len(normalized_clauses) > 2
        for key, ascending, is_group, source_index in effective_clauses:
            chip = QToolButton()
            direction = "ASC" if ascending else "DESC"
            suffix = "  x" if can_remove and not is_group and source_index is not None else ""
            prefix = "Group: " if is_group else ""
            chip.setText(f"{prefix}{SORT_LABELS.get(key, key)} {direction}{suffix}")
            chip.setAutoRaise(True)
            if can_remove and not is_group and source_index is not None:
                chip.clicked.connect(
                    lambda _checked=False, clause_index=source_index: (
                        self._remove_sort_clause(clause_index)
                    )
                )
            else:
                chip.setEnabled(False)
            self.sort_chip_layout.insertWidget(self.sort_chip_layout.count() - 1, chip)
        if not any(key == "path_id" for key, _ascending, _group, _index in effective_clauses):
            tie_chip = QToolButton()
            tie_chip.setText("Path ID ASC (tie)")
            tie_chip.setToolTip("Canonical Path ID is always the final deterministic tie-breaker.")
            tie_chip.setAutoRaise(True)
            tie_chip.setEnabled(False)
            self.sort_chip_layout.insertWidget(self.sort_chip_layout.count() - 1, tie_chip)

    def _rebuild_column_menu(self) -> None:
        self._column_menu.clear()
        model = self.table.model()
        if model is None:
            empty = self._column_menu.addAction("No frame loaded")
            empty.setEnabled(False)
            return
        for column in range(model.columnCount()):
            label = model.headerData(column, Qt.Horizontal, Qt.DisplayRole)
            action = self._column_menu.addAction(str(label or f"Column {column + 1}"))
            action.setCheckable(True)
            action.setChecked(not self.table.isColumnHidden(column))
            action.toggled.connect(
                lambda visible, column_index=column: self._set_column_visible(
                    column_index,
                    visible,
                )
            )

    def _set_column_visible(self, column: int, visible: bool) -> None:
        """Toggle one column and request worker-side preparation when needed."""
        model = self.table.model()
        if model is None:
            return
        self.table.setColumnHidden(int(column), not bool(visible))
        columns = getattr(model, "columns", ())
        try:
            column_key = getattr(columns[int(column)], "value", columns[int(column)])
        except (IndexError, TypeError):
            return
        normalized = str(column_key)
        if self._optional_start_column is not None and int(column) >= int(
            self._optional_start_column
        ):
            if visible:
                self._visible_optional_columns.add(normalized)
            else:
                self._visible_optional_columns.discard(normalized)
        self.columnVisibilityChanged.emit(normalized, bool(visible))

    def _set_filter_drawer_visible(self, visible: bool) -> None:
        self.filter_drawer.setVisible(bool(visible))

    def _apply_filters(self) -> None:
        """Parse filter text and emit one immutable-value mapping."""
        try:
            filters = {
                "tx_ids": self._parse_int_tuple(self.filter_edits["tx_ids"].text()),
                "rx_ids": self._parse_int_tuple(self.filter_edits["rx_ids"].text()),
                "path_loss_min_db": self._parse_optional_float(
                    self.filter_edits["path_loss_min_db"].text()
                ),
                "path_loss_max_db": self._parse_optional_float(
                    self.filter_edits["path_loss_max_db"].text()
                ),
                "delay_min_ns": self._parse_optional_float(
                    self.filter_edits["delay_min_ns"].text()
                ),
                "delay_max_ns": self._parse_optional_float(
                    self.filter_edits["delay_max_ns"].text()
                ),
                "interaction_count_min": self._parse_optional_int(
                    self.filter_edits["interaction_count_min"].text()
                ),
                "interaction_count_max": self._parse_optional_int(
                    self.filter_edits["interaction_count_max"].text()
                ),
                "contains_interactions": self._parse_int_tuple(
                    self.filter_edits["contains_interactions"].text()
                ),
                "pure_interaction": self._parse_optional_int(
                    self.filter_edits["pure_interaction"].text()
                ),
                "mixed_only": self.mixed_only_checkbox.isChecked(),
                "exact_interaction_sequence": self._parse_optional_int_tuple(
                    self.filter_edits["exact_interaction_sequence"].text()
                ),
                "first_material_ids": self._parse_int_tuple(
                    self.filter_edits["first_material_ids"].text()
                ),
            }
        except ValueError as exc:
            self.set_status(f"Invalid filter: {exc}")
            return
        active_count = sum(value not in (None, (), False) for value in filters.values())
        self.filter_button.setText(f"Filters ({active_count})" if active_count else "Filters")
        self.filtersChanged.emit(filters)

    def _clear_filters(self) -> None:
        for edit in self.filter_edits.values():
            edit.clear()
        self.mixed_only_checkbox.setChecked(False)
        self.filter_button.setText("Filters")
        self._apply_filters()

    @staticmethod
    def _split_values(text: str) -> tuple[str, ...]:
        return tuple(value.strip() for value in str(text).split(",") if value.strip())

    @classmethod
    def _parse_int_tuple(cls, text: str) -> tuple[int, ...]:
        return tuple(int(value) for value in cls._split_values(text))

    @classmethod
    def _parse_optional_int_tuple(cls, text: str) -> tuple[int, ...] | None:
        values = cls._split_values(text)
        return tuple(int(value) for value in values) if values else None

    @staticmethod
    def _parse_optional_int(text: str) -> int | None:
        stripped = str(text).strip()
        return int(stripped) if stripped else None

    @staticmethod
    def _parse_optional_float(text: str) -> float | None:
        stripped = str(text).strip()
        return float(stripped) if stripped else None

    def _on_current_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        if self._programmatic_path_selection or not current.isValid():
            return
        model = current.model()
        path_id = self._model_path_id_for_row(model, current.row())
        if path_id is not None:
            self.pathSelectionRequested.emit(path_id)

    @staticmethod
    def _model_path_id_for_row(model: Any, row: int) -> int | None:
        for name in ("path_id_for_row", "path_id_at"):
            resolver = getattr(model, name, None)
            if callable(resolver):
                try:
                    return int(resolver(row))
                except (IndexError, TypeError, ValueError):
                    return None
        index = model.index(row, 0)
        for role in (Qt.UserRole, Qt.DisplayRole):
            value = model.data(index, role)
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _model_row_for_path(model: Any, path_id: int) -> int | None:
        for name in ("row_for_path_id", "find_row"):
            resolver = getattr(model, name, None)
            if callable(resolver):
                try:
                    value = resolver(path_id)
                except (IndexError, TypeError, ValueError):
                    return None
                return None if value is None else int(value)
        return None

    def _set_activity(self, active: bool) -> None:
        active = bool(active)
        if active == self._activity_state:
            return
        self._activity_state = active
        self.activityChanged.emit(active)

    def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self._set_activity(not self.isMinimized())

    def hideEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        self._set_activity(False)
        super().hideEvent(event)

    def changeEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange and self.isVisible():
            self._set_activity(not self.isMinimized())

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        self._set_activity(False)
        self.closing.emit()
        super().closeEvent(event)
