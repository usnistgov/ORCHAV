"""Node, target, trajectory-style, and temporary actor-edit controls.

The panel owns widget construction for TX/RX/target selection and presentation
settings. Node population, label semantics, movement previews, and renderer
updates are handled by services and controllers that consume the registered
widget keys.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..scene.defaults import (
    DEFAULT_LABEL_FONT_SIZE,
    DEFAULT_LABEL_OFFSET_M,
    DEFAULT_NODE_MARKER_SIZE_M,
    DEFAULT_ORIENTATION_SCALE_M,
    DEFAULT_TRAJECTORY_LINE_WIDTH_PX,
    DEFAULT_TRAJECTORY_POINT_SIZE_PX,
    LABEL_FONT_SIZE_BOUNDS,
    LABEL_OFFSET_BOUNDS_M,
    NODE_MARKER_SIZE_BOUNDS_M,
    ORIENTATION_SCALE_BOUNDS_M,
    TRAJECTORY_LINE_WIDTH_BOUNDS_PX,
    TRAJECTORY_POINT_SIZE_BOUNDS_PX,
)
from ..utils.colors import ensure_viridis_lut
from .base import BasePanel
from .data_source.raytracing_section import RaytracingControlSection
from .ui_theme import compact_button_style, configure_label


def _viridis_gradient_pixmap(width: int = 200, height: int = 14) -> QPixmap:
    """Create a trajectory gradient from the canonical viridis LUT."""
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    gradient = QLinearGradient(0, 0, width, 0)
    lut = ensure_viridis_lut()
    indices = np.linspace(0, len(lut) - 1, 18).astype(np.int32)
    denominator = max(len(indices) - 1, 1)
    for stop_index, lut_index in enumerate(indices):
        color = np.clip(lut[lut_index, :3], 0.0, 1.0)
        gradient.setColorAt(
            float(stop_index) / float(denominator),
            QColor.fromRgbF(
                float(color[0]),
                float(color[1]),
                float(color[2]),
            ),
        )
    painter.fillRect(0, 0, width, height, gradient)
    painter.end()
    return pixmap


class NodesSelectionPanel(BasePanel):
    """Build the Nodes-tab control surface for communication nodes and targets."""

    def __init__(self, parent_widget):
        """Initialize the shared widget registry from ``BasePanel``."""
        super().__init__(parent_widget)

    def create_panel(self):
        """Create grouped TX/RX/target controls and shared display settings.

        Interactive editing is exposed separately through
        :meth:`create_interactive_preview_panel` so the workflow can live in a
        capability-gated tab without duplicating its widgets in Scene.
        """
        group = self.create_group_box("Nodes")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        grid.setContentsMargins(0, 0, 0, 0)

        self._entity_grid = grid

        # Row 0: entity-level controls
        grid.addWidget(self._create_tx_group(), 0, 0)
        grid.addWidget(self._create_rx_group(), 0, 1)
        self._target_group = self._create_target_group()
        grid.addWidget(self._target_group, 0, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)

        # Rows below: shared styling and display controls
        grid.addWidget(self._create_display_group(), 1, 0, 1, 3)
        grid.addWidget(self._create_coloring_group(), 2, 0, 1, 3)
        self._trajectory_style_group = self._create_trajectory_style_group()
        grid.addWidget(self._trajectory_style_group, 3, 0, 1, 3)
        layout.addLayout(grid)
        self.update_node_rename_visibility()
        self.update_target_controls()
        self.update_trajectory_style_enabled()
        return group

    def create_interactive_preview_panel(self) -> QGroupBox:
        """Create the source-aware temporary actor editing controls."""
        group = self.create_group_box("Edit")
        self._interactive_edit_panel = group
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)
        mode_label = QLabel("Open a scenario to see the available edit workflow.")
        configure_label(mode_label, role="muted", font_size=10, word_wrap=True)
        self.widgets["interactive_edit_mode_label"] = mode_label
        layout.addWidget(mode_label)
        layout.addWidget(self._create_interactive_preview_row())
        self.refresh_live_preview_state()
        return group

    # --- group builders -------------------------------------------------

    def _create_tx_group(self) -> QGroupBox:
        """Create transmitter display controls and the contextual rename action."""
        return self._create_tx_rx_group(
            title="TX",
            prefix="tx",
            size_tooltip="Transmitter marker size in meters",
            rename_tooltip="Rename the selected transmitter node",
        )

    def _create_rx_group(self) -> QGroupBox:
        """Create receiver display controls and the contextual rename action."""
        return self._create_tx_rx_group(
            title="RX",
            prefix="rx",
            size_tooltip="Receiver marker size in meters",
            rename_tooltip="Rename the selected receiver node",
        )

    def _create_tx_rx_group(
        self,
        *,
        title: str,
        prefix: str,
        size_tooltip: str,
        rename_tooltip: str,
    ) -> QGroupBox:
        """Create a TX/RX display subgroup with stable widget keys by prefix."""
        box = QGroupBox(title)
        box.setStyleSheet("QGroupBox { font-size: 10px; }")
        vbox = QVBoxLayout(box)
        vbox.setSpacing(4)
        vbox.setContentsMargins(4, 8, 4, 4)

        # TX/RX scope is owned by the persistent Context strip. This row keeps
        # only the appearance setting that is local to the Scene workflow.
        row1 = QHBoxLayout()
        row1.setSpacing(4)
        row1.addWidget(QLabel("Size:"))
        size_spin = QDoubleSpinBox()
        size_spin.setRange(*NODE_MARKER_SIZE_BOUNDS_M)
        size_spin.setSingleStep(0.1)
        size_spin.setValue(
            float(getattr(self.parent, f"{prefix}_marker_size", DEFAULT_NODE_MARKER_SIZE_M))
        )
        size_spin.setDecimals(3)
        size_spin.setSuffix(" m")
        size_spin.setToolTip(size_tooltip)
        size_spin.setMinimumWidth(86)
        self.widgets[f"{prefix}_marker_size_spin"] = size_spin
        row1.addWidget(size_spin)
        row1.addStretch()
        vbox.addLayout(row1)

        # Row 2: per-node display toggles
        row2 = QHBoxLayout()
        row2.setSpacing(4)

        orient_cb = QCheckBox("Axes")
        orient_cb.setChecked(False)
        orient_cb.setToolTip(f"Show orientation frame (XYZ axes) for {title} nodes")
        self.widgets[f"{prefix}_orient_cb"] = orient_cb
        row2.addWidget(orient_cb)

        trajectory_cb = QCheckBox("Trajectory")
        trajectory_cb.setChecked(False)
        trajectory_cb.setToolTip(f"Show full {title} trajectory as 3D lines in the scene")
        trajectory_cb.toggled.connect(self.update_trajectory_style_enabled)
        self.widgets[f"{prefix}_trajectory_cb"] = trajectory_cb
        row2.addWidget(trajectory_cb)

        row2.addStretch()
        vbox.addLayout(row2)

        # Row 3: rename action
        row3 = QHBoxLayout()
        row3.setSpacing(4)
        rename_btn = QPushButton("Rename")
        rename_btn.setFixedWidth(80)
        rename_btn.setToolTip(rename_tooltip)
        rename_btn.clicked.connect(lambda: self._rename_node(title))
        rename_btn.setVisible(False)
        self.widgets[f"{prefix}_rename_btn"] = rename_btn
        row3.addWidget(rename_btn)

        row3.addStretch()
        vbox.addLayout(row3)
        return box

    def _create_target_group(self) -> QGroupBox:
        """Create controls for target visibility, labels, trajectories, and rename."""
        box = QGroupBox("Targets")
        box.setStyleSheet("QGroupBox { font-size: 10px; }")
        vbox = QVBoxLayout(box)
        vbox.setSpacing(4)
        vbox.setContentsMargins(4, 8, 4, 4)

        # Row 1: target selector + visibility
        row1 = QHBoxLayout()
        row1.setSpacing(4)

        self.widgets["target_dropdown"] = QComboBox()
        self.widgets["target_dropdown"].setMinimumWidth(105)
        self.widgets["target_dropdown"].setStyleSheet("QComboBox { padding: 2px; }")
        self.widgets["target_dropdown"].setToolTip("Select a target to rename")
        self.widgets["target_dropdown"].currentIndexChanged.connect(
            lambda _idx: self._update_target_rename_visibility()
        )
        row1.addWidget(self.widgets["target_dropdown"], stretch=1)

        self.widgets["target_cb"] = QCheckBox("Show")
        self.widgets["target_cb"].setChecked(True)
        self.widgets["target_cb"].setToolTip("Show target objects in the visualization")
        row1.addWidget(self.widgets["target_cb"])
        vbox.addLayout(row1)

        # Row 2: target display toggles
        row2 = QHBoxLayout()
        row2.setSpacing(4)

        self.widgets["target_orient_cb"] = QCheckBox("Axes")
        self.widgets["target_orient_cb"].setChecked(False)
        self.widgets["target_orient_cb"].setToolTip(
            "Show orientation frame (XYZ axes) for target objects"
        )
        row2.addWidget(self.widgets["target_orient_cb"])

        self.widgets["target_trajectory_cb"] = QCheckBox("Trajectory")
        self.widgets["target_trajectory_cb"].setChecked(False)
        self.widgets["target_trajectory_cb"].setToolTip(
            "Show full target trajectory as 3D lines in the scene"
        )
        self.widgets["target_trajectory_cb"].toggled.connect(self.update_trajectory_style_enabled)
        row2.addWidget(self.widgets["target_trajectory_cb"])

        row2.addStretch()
        vbox.addLayout(row2)

        # Row 3: rename action
        row3 = QHBoxLayout()
        row3.setSpacing(4)
        self.widgets["target_rename_btn"] = QPushButton("Rename")
        self.widgets["target_rename_btn"].setFixedWidth(80)
        self.widgets["target_rename_btn"].setToolTip("Rename the selected target")
        self.widgets["target_rename_btn"].clicked.connect(self._rename_target)
        self.widgets["target_rename_btn"].setVisible(False)
        row3.addWidget(self.widgets["target_rename_btn"])

        row3.addStretch()
        vbox.addLayout(row3)
        return box

    def _create_display_group(self) -> QGroupBox:
        """Create shared label, marker-size, axis-size, and offset controls."""
        box = QGroupBox("Display")
        box.setStyleSheet("QGroupBox { font-size: 10px; }")
        vbox = QVBoxLayout(box)
        vbox.setSpacing(4)
        vbox.setContentsMargins(4, 8, 4, 4)

        # Row 1: label toggles + global sizes
        row1 = QHBoxLayout()
        row1.setSpacing(4)
        self.widgets["labels_cb"] = QCheckBox("TX/RX labels")
        self.widgets["labels_cb"].setChecked(True)
        self.widgets["labels_cb"].setToolTip(
            "Display text labels for transmitter and receiver nodes"
        )
        row1.addWidget(self.widgets["labels_cb"])

        row1.addWidget(QLabel("TX/RX text:"))
        self.widgets["node_label_mode_combo"] = QComboBox()
        self.widgets["node_label_mode_combo"].addItem("Role", "role")
        self.widgets["node_label_mode_combo"].addItem("Device Name", "name")
        self.widgets["node_label_mode_combo"].setToolTip(
            "Choose whether transmitter/receiver labels use TX1/RX1 roles or scenario device names"
        )
        self.widgets["node_label_mode_combo"].setFixedWidth(118)
        row1.addWidget(self.widgets["node_label_mode_combo"])

        self.widgets["target_labels_cb"] = QCheckBox("Target labels")
        self.widgets["target_labels_cb"].setChecked(True)
        self.widgets["target_labels_cb"].setToolTip("Display text labels for target objects")
        row1.addWidget(self.widgets["target_labels_cb"])

        row1.addWidget(QLabel("Label font:"))
        self.widgets["label_font_size_spin"] = QDoubleSpinBox()
        self.widgets["label_font_size_spin"].setRange(*LABEL_FONT_SIZE_BOUNDS)
        self.widgets["label_font_size_spin"].setSingleStep(0.1)
        self.widgets["label_font_size_spin"].setValue(
            float(getattr(self.parent, "label_font_size", DEFAULT_LABEL_FONT_SIZE))
        )
        self.widgets["label_font_size_spin"].setDecimals(2)
        self.widgets["label_font_size_spin"].setToolTip("Font size for node and target labels")
        self.widgets["label_font_size_spin"].setMinimumWidth(76)
        row1.addWidget(self.widgets["label_font_size_spin"])

        row1.addWidget(QLabel("Axis size:"))
        self.widgets["orientation_scale_spin"] = QDoubleSpinBox()
        self.widgets["orientation_scale_spin"].setRange(*ORIENTATION_SCALE_BOUNDS_M)
        self.widgets["orientation_scale_spin"].setSingleStep(0.1)
        self.widgets["orientation_scale_spin"].setValue(
            float(getattr(self.parent, "orientation_scale", DEFAULT_ORIENTATION_SCALE_M))
        )
        self.widgets["orientation_scale_spin"].setDecimals(3)
        self.widgets["orientation_scale_spin"].setSuffix(" m")
        self.widgets["orientation_scale_spin"].setToolTip("Orientation-frame axis size in meters")
        self.widgets["orientation_scale_spin"].setMinimumWidth(86)
        row1.addWidget(self.widgets["orientation_scale_spin"])
        row1.addStretch()
        vbox.addLayout(row1)

        # Row 2: label offset X/Y/Z spinboxes
        row2 = QHBoxLayout()
        row2.setSpacing(3)
        row2.addWidget(QLabel("Label offset:"))
        for axis, default in zip(("x", "y", "z"), DEFAULT_LABEL_OFFSET_M):
            row2.addWidget(QLabel(f"{axis.upper()}:"))
            spin = QDoubleSpinBox()
            spin.setRange(*LABEL_OFFSET_BOUNDS_M)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            spin.setValue(float(getattr(self.parent, f"label_offset_{axis}", default)))
            spin.setSuffix(" m")
            spin.setToolTip(f"Label offset along the {axis.upper()} axis in meters")
            spin.setMinimumWidth(92)
            self.widgets[f"{axis}_offset_spinbox"] = spin
            row2.addWidget(spin)
        row2.addStretch()
        vbox.addLayout(row2)

        return box

    def _create_coloring_group(self) -> QGroupBox:
        """Create node coloring-mode controls and TX/RX legend labels."""
        box = QGroupBox("Coloring")
        box.setStyleSheet("QGroupBox { font-size: 10px; }")
        row = QHBoxLayout(box)
        row.setSpacing(8)
        row.setContentsMargins(4, 8, 4, 4)

        # Radio buttons
        self.widgets["node_coloring_group"] = QButtonGroup()
        self.widgets["per_node_type_rb"] = QRadioButton("Per Node Type")
        self.widgets["per_node_type_rb"].setChecked(True)
        self.widgets["per_node_type_rb"].setToolTip(
            "Color all transmitters one color and all receivers another color"
        )
        self.widgets["node_coloring_group"].addButton(self.widgets["per_node_type_rb"])
        row.addWidget(self.widgets["per_node_type_rb"])

        self.widgets["individual_nodes_rb"] = QRadioButton("Individual Nodes")
        self.widgets["individual_nodes_rb"].setToolTip(
            "Assign a unique color to each individual node for easy identification"
        )
        self.widgets["node_coloring_group"].addButton(self.widgets["individual_nodes_rb"])
        row.addWidget(self.widgets["individual_nodes_rb"])

        # Legend labels
        legend_container = QWidget()
        legend_layout = QVBoxLayout(legend_container)
        legend_layout.setSpacing(1)
        legend_layout.setContentsMargins(0, 0, 0, 0)
        self.widgets["tx_rx_legend_layout"] = legend_layout
        self.widgets["tx_legend_label"] = QLabel("Transmitters")
        self.widgets["tx_legend_label"].setStyleSheet("color: red; font-size: 10px;")
        legend_layout.addWidget(self.widgets["tx_legend_label"])
        self.widgets["rx_legend_label"] = QLabel("Receivers")
        self.widgets["rx_legend_label"].setStyleSheet("color: blue; font-size: 10px;")
        legend_layout.addWidget(self.widgets["rx_legend_label"])
        row.addWidget(legend_container, stretch=1)
        return box

    def _create_trajectory_style_group(self) -> QGroupBox:
        """Create scalar trajectory color and geometry-size controls."""
        box = QGroupBox("Trajectory Style")
        box.setStyleSheet("QGroupBox { font-size: 10px; }")
        vbox = QVBoxLayout(box)
        vbox.setSpacing(4)
        vbox.setContentsMargins(4, 8, 4, 4)

        # Row 1: trajectory color mode + status
        row1 = QHBoxLayout()
        row1.setSpacing(4)
        row1.addWidget(QLabel("Color:"))

        self.widgets["trajectory_color_group"] = QButtonGroup()
        for mode_id, label in [
            ("node_color", "Node"),
            ("speed", "Speed"),
            ("altitude", "Altitude"),
            ("time", "Time"),
            ("angular_speed", "Ang.Speed"),
        ]:
            rb = QRadioButton(label)
            rb.setChecked(mode_id == "node_color")
            rb.setToolTip(f"Color trajectory by {label.lower()}")
            rb.setProperty("trajectory_color_mode", mode_id)
            self.widgets[f"trajectory_color_{mode_id}_rb"] = rb
            self.widgets["trajectory_color_group"].addButton(rb)
            row1.addWidget(rb)

        self.widgets["trajectory_status_label"] = QLabel("")
        configure_label(
            self.widgets["trajectory_status_label"],
            role="muted",
            font_size=10,
            italic=True,
        )
        row1.addWidget(self.widgets["trajectory_status_label"])

        row1.addStretch()
        vbox.addLayout(row1)

        # Row 2: Viridis colorbar with min/max labels (hidden when "Node" is active)
        colorbar_container = QWidget()
        colorbar_layout = QHBoxLayout(colorbar_container)
        colorbar_layout.setSpacing(4)
        colorbar_layout.setContentsMargins(24, 0, 0, 0)  # indent under "Color:" label

        self.widgets["trajectory_colorbar_min_label"] = QLabel("")
        configure_label(
            self.widgets["trajectory_colorbar_min_label"],
            role="muted",
            font_size=9,
        )
        self.widgets["trajectory_colorbar_min_label"].setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.widgets["trajectory_colorbar_min_label"].setFixedWidth(50)
        colorbar_layout.addWidget(self.widgets["trajectory_colorbar_min_label"])

        gradient_label = QLabel()
        gradient_label.setPixmap(_viridis_gradient_pixmap(160, 12))
        gradient_label.setFixedSize(160, 12)
        gradient_label.setToolTip("Viridis colormap — maps scalar values to color")
        self.widgets["trajectory_colorbar_gradient"] = gradient_label
        colorbar_layout.addWidget(gradient_label)

        self.widgets["trajectory_colorbar_max_label"] = QLabel("")
        configure_label(
            self.widgets["trajectory_colorbar_max_label"],
            role="muted",
            font_size=9,
        )
        self.widgets["trajectory_colorbar_max_label"].setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.widgets["trajectory_colorbar_max_label"].setFixedWidth(50)
        colorbar_layout.addWidget(self.widgets["trajectory_colorbar_max_label"])

        colorbar_layout.addStretch()
        self.widgets["trajectory_colorbar_container"] = colorbar_container
        colorbar_container.setVisible(False)  # hidden by default (Node mode)
        vbox.addWidget(colorbar_container)

        # Row 3: Line width + point size controls
        row3 = QHBoxLayout()
        row3.setSpacing(4)

        row3.addWidget(QLabel("Line:"))
        self.widgets["trajectory_line_width_spin"] = QDoubleSpinBox()
        self.widgets["trajectory_line_width_spin"].setRange(*TRAJECTORY_LINE_WIDTH_BOUNDS_PX)
        self.widgets["trajectory_line_width_spin"].setSingleStep(0.5)
        renderer = getattr(self.parent, "renderer", None)
        self.widgets["trajectory_line_width_spin"].setValue(
            float(
                getattr(
                    renderer,
                    "trajectory_line_width",
                    DEFAULT_TRAJECTORY_LINE_WIDTH_PX,
                )
            )
        )
        self.widgets["trajectory_line_width_spin"].setDecimals(1)
        self.widgets["trajectory_line_width_spin"].setSuffix(" px")
        self.widgets["trajectory_line_width_spin"].setToolTip("Trajectory line width in pixels")
        self.widgets["trajectory_line_width_spin"].setMinimumWidth(82)
        row3.addWidget(self.widgets["trajectory_line_width_spin"])

        row3.addWidget(QLabel("Point:"))
        self.widgets["trajectory_point_size_spin"] = QDoubleSpinBox()
        self.widgets["trajectory_point_size_spin"].setRange(*TRAJECTORY_POINT_SIZE_BOUNDS_PX)
        self.widgets["trajectory_point_size_spin"].setSingleStep(1.0)
        self.widgets["trajectory_point_size_spin"].setValue(
            float(
                getattr(
                    renderer,
                    "trajectory_point_size",
                    DEFAULT_TRAJECTORY_POINT_SIZE_PX,
                )
            )
        )
        self.widgets["trajectory_point_size_spin"].setDecimals(1)
        self.widgets["trajectory_point_size_spin"].setSuffix(" px")
        self.widgets["trajectory_point_size_spin"].setToolTip("Trajectory point size in pixels")
        self.widgets["trajectory_point_size_spin"].setMinimumWidth(82)
        row3.addWidget(self.widgets["trajectory_point_size_spin"])

        row3.addStretch()
        vbox.addLayout(row3)

        return box

    def _create_live_preview_group(self) -> QGroupBox:
        """Create pygfx controls shared by local and live actor editing."""
        box = QGroupBox("Actor Editing")
        box.setStyleSheet("QGroupBox { font-size: 10px; }")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 8, 4, 4)

        cb = QCheckBox("Enable Actor Editing")
        cb.setToolTip("Use the pygfx gizmo to move TX/RX nodes or targets")
        self.widgets["live_preview_cb"] = cb

        primary_row = QHBoxLayout()
        primary_row.setSpacing(6)
        primary_row.addWidget(cb)

        recompute_btn = QPushButton("Recompute")
        recompute_btn.setToolTip(
            "Run a final-quality preview solve with the current edited actor poses"
        )
        recompute_btn.setFixedWidth(96)
        self.widgets["live_preview_recompute_btn"] = recompute_btn
        primary_row.addWidget(recompute_btn)
        primary_row.addStretch()
        layout.addLayout(primary_row)

        reset_selected_btn = QPushButton("Reset Selected")
        reset_selected_btn.setToolTip("Reset the selected edited TX/RX/target to its loaded pose")
        reset_selected_btn.setFixedWidth(112)
        self.widgets["live_preview_reset_selected_btn"] = reset_selected_btn

        reset_all_btn = QPushButton("Reset All")
        reset_all_btn.setToolTip("Reset all transient interactive edits")
        reset_all_btn.setFixedWidth(82)
        self.widgets["live_preview_reset_all_btn"] = reset_all_btn

        reset_row = QHBoxLayout()
        reset_row.setSpacing(6)
        reset_row.addWidget(reset_selected_btn)
        reset_row.addWidget(reset_all_btn)
        reset_row.addStretch()
        layout.addLayout(reset_row)

        status = QLabel("Editing off")
        configure_label(status, role="muted", font_size=10, italic=True)
        status.setWordWrap(True)
        status.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        status.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.widgets["live_preview_status_label"] = status
        layout.addWidget(status)
        return box

    def _create_interactive_preview_row(self) -> QWidget:
        """Create the live-preview row with adjacent solver settings."""
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setSpacing(6)
        row.setContentsMargins(0, 0, 0, 0)
        self._live_preview_group = self._create_live_preview_group()
        self._preview_raytracing_group = self._create_preview_raytracing_group()
        row.addWidget(self._live_preview_group, stretch=1)
        row.addWidget(self._preview_raytracing_group, stretch=2)
        return row_widget

    def _create_preview_raytracing_group(self) -> QWidget:
        """Create raytracing settings shared by local preview and live editing."""
        self._preview_raytracing_section = RaytracingControlSection(
            self.parent,
            self.widgets,
            compact_button_style,
            group_title="Raytracing",
            show_apply_button=True,
            status_text="Available after a scenario is loaded",
            initial_preset="ultra-low",
            sync_initial_preset=False,
        )
        return self._preview_raytracing_section.create_content()

    # --- visibility helpers -------------------------------------------------

    def _update_node_rename_visibility(self, prefix: str) -> None:
        """Show rename for a concrete global TX/RX selection."""
        rename_btn = self.widgets.get(f"{prefix}_rename_btn")
        if rename_btn is None:
            return
        state = getattr(self.parent, "app_state", None)
        selection = getattr(state, f"selected_{prefix}", "all")
        has_specific_selection = selection not in {"all", None}
        rename_btn.setVisible(has_specific_selection)

    def _update_target_rename_visibility(self) -> None:
        """Show target rename only when a concrete target is selected."""
        dropdown = self.widgets.get("target_dropdown")
        rename_btn = self.widgets.get("target_rename_btn")
        if dropdown is None or rename_btn is None:
            return

        rename_btn.setVisible(dropdown.count() > 0 and dropdown.currentData() is not None)

    def update_node_rename_visibility(self) -> None:
        """Refresh rename buttons after dropdown contents or selection change."""
        self._update_node_rename_visibility("tx")
        self._update_node_rename_visibility("rx")
        self._update_target_rename_visibility()

    def update_target_controls(self) -> None:
        """Show target controls only when target entries exist."""
        target_entries = getattr(self.parent, "target_entries", [])
        has_targets = bool(target_entries)

        target_group = getattr(self, "_target_group", None)
        if target_group is not None:
            target_group.setVisible(has_targets)

        target_labels_cb = self.widgets.get("target_labels_cb")
        if target_labels_cb is not None:
            target_labels_cb.setVisible(has_targets)

        grid = getattr(self, "_entity_grid", None)
        if grid is not None:
            grid.setColumnStretch(2, 1 if has_targets else 0)

        self._update_target_rename_visibility()
        self.update_trajectory_style_enabled()

    def update_trajectory_style_enabled(self, _checked: bool | None = None) -> None:
        """Enable trajectory styling only while a visible trajectory is active."""
        trajectory_group = getattr(self, "_trajectory_style_group", None)
        if trajectory_group is None:
            return

        tx_cb = self.widgets.get("tx_trajectory_cb")
        rx_cb = self.widgets.get("rx_trajectory_cb")
        tx_enabled = bool(tx_cb is not None and tx_cb.isChecked())
        rx_enabled = bool(rx_cb is not None and rx_cb.isChecked())

        target_group = getattr(self, "_target_group", None)
        target_visible = target_group is not None and not target_group.isHidden()
        target_cb = self.widgets.get("target_trajectory_cb")
        target_enabled = bool(target_visible and target_cb is not None and target_cb.isChecked())

        trajectory_group.setEnabled(tx_enabled or rx_enabled or target_enabled)

    def refresh_live_preview_state(self) -> None:
        """Refresh the Edit tab from the active frame-source policy."""
        group = getattr(self, "_live_preview_group", None)
        if group is None:
            return
        service = getattr(self.parent, "live_preview_service", None)
        mode = (
            service.edit_mode() if callable(getattr(service, "edit_mode", None)) else "unavailable"
        )
        available = bool(service is not None and service.is_available())
        actor_mode = mode in {"files", "live_grpc"}
        group.setVisible(actor_mode)
        group.setEnabled(available)

        mode_label = self.widgets.get("interactive_edit_mode_label")
        if mode_label is not None:
            if mode == "files":
                message = (
                    "Local what-if preview. Actor changes affect the current preview only; "
                    "scenario files and generated frames are unchanged."
                )
                if not available:
                    message += " Actor gizmos require the pygfx renderer."
            elif mode == "live_grpc":
                message = (
                    "Live session edits. Actor and raytracing changes are sent to the connected "
                    "generator and are not saved to scenario YAML."
                )
                if not available:
                    message += " Actor gizmos require pygfx; raytracing controls remain available."
            elif mode == "remote_hdf5":
                message = "Remote HDF5 frame sets are read-only."
            else:
                message = "Open a scenario to see the available edit workflow."
            mode_label.setText(message)

        preview_rt_group = getattr(self, "_preview_raytracing_group", None)
        if preview_rt_group is not None:
            preview_rt_group.setVisible(mode == "live_grpc" or (mode == "files" and available))

        apply_btn = self.widgets.get("rt_apply_btn")
        if apply_btn is not None:
            apply_btn.setVisible(mode == "live_grpc")
        rt_status = self.widgets.get("rt_status_label")
        if rt_status is not None:
            rt_status.setText(
                "Applied to the connected generator"
                if mode == "live_grpc"
                else "Used by the local what-if preview"
            )
        self._sync_raytracing_from_active_scenario(mode)

        cb = self.widgets.get("live_preview_cb")
        if cb is not None:
            cb.blockSignals(True)
            cb.setChecked(bool(getattr(service, "enabled", False)) if available else False)
            cb.setEnabled(available)
            cb.setText(
                "Enable Live Actor Editing"
                if mode == "live_grpc"
                else "Enable Local What-if Preview"
            )
            cb.blockSignals(False)

        btn = self.widgets.get("live_preview_recompute_btn")
        if btn is not None:
            btn.setVisible(mode == "files")
            btn.setEnabled(bool(available and getattr(service, "enabled", False)))
        dirty = bool(
            available
            and getattr(service, "enabled", False)
            and callable(getattr(service, "dirty_edit_count", None))
            and service.dirty_edit_count() > 0
        )
        reset_selected = self.widgets.get("live_preview_reset_selected_btn")
        if reset_selected is not None:
            reset_selected.setEnabled(dirty)
        reset_all = self.widgets.get("live_preview_reset_all_btn")
        if reset_all is not None:
            reset_all.setEnabled(dirty)

        default_status = "Live session editing off" if mode == "live_grpc" else "Local preview off"
        status = str(getattr(self.parent, "_live_preview_status", default_status))
        if status in {"Preview off", "Editing off"}:
            status = default_status
        self.update_live_preview_status(status)

    def _sync_raytracing_from_active_scenario(self, mode: str) -> None:
        """Initialize one scenario's edit controls without overwriting user changes."""
        scenario = getattr(self.parent, "scenario", None)
        if scenario is None or mode not in {"files", "live_grpc"}:
            return
        identity = (id(scenario), mode)
        if getattr(self, "_raytracing_scenario_identity", None) == identity:
            return
        sync = getattr(self._preview_raytracing_section, "sync_from_scenario", None)
        if callable(sync):
            sync(scenario)
        self._raytracing_scenario_identity = identity

    def update_live_preview_status(self, message: str) -> None:
        """Update the interactive preview status label."""
        label = self.widgets.get("live_preview_status_label")
        if label is not None:
            label.setText(str(message))

    # --- renaming helpers ---------------------------------------------------

    def _rename_node(self, node_type: str) -> None:
        """Prompt the user to rename the currently selected TX or RX node.

        Args:
            node_type: ``"TX"`` or ``"RX"``.
        """
        prefix = node_type.lower()
        state = getattr(self.parent, "app_state", None)
        node_idx = getattr(state, f"selected_{prefix}", "all")
        if node_idx in {"all", None}:
            return
        dropdown = getattr(self.parent, f"{prefix}_dropdown", None)
        current_text = (
            dropdown.currentText() if dropdown is not None else f"{node_type}{int(node_idx) + 1}"
        )
        new_label, ok = QInputDialog.getText(
            None,
            f"Rename {node_type} Node",
            f"New label for {current_text}:",
            text=current_text,
        )
        if not ok or not new_label.strip():
            return

        new_label = new_label.strip()

        # Propagate the rename to the controller / visualizer
        viz = self.parent
        if hasattr(viz, "ui_controller") and hasattr(viz.ui_controller, "handle_node_renamed"):
            viz.ui_controller.handle_node_renamed(node_type, node_idx, new_label)

    def _rename_target(self) -> None:
        """Prompt the user to rename the currently selected target."""
        dropdown = self.widgets.get("target_dropdown")
        if dropdown is None or dropdown.count() == 0:
            return

        target_idx = dropdown.currentData()
        if target_idx is None:
            return

        current_text = dropdown.currentText()
        new_label, ok = QInputDialog.getText(
            None,
            "Rename Target",
            f"New label for {current_text}:",
            text=current_text,
        )
        if not ok or not new_label.strip():
            return

        new_label = new_label.strip()
        viz = self.parent
        if hasattr(viz, "ui_controller") and hasattr(viz.ui_controller, "handle_target_renamed"):
            viz.ui_controller.handle_target_renamed(target_idx, new_label)

    def populate_target_dropdown(self) -> None:
        """Populate the target dropdown with current target names."""
        dropdown = self.widgets.get("target_dropdown")
        if dropdown is None:
            return

        dropdown.blockSignals(True)
        dropdown.clear()

        viz = self.parent
        target_entries = getattr(viz, "target_entries", [])
        custom_labels = getattr(getattr(viz, "app_state", None), "target_labels", ())

        for i, entry in enumerate(target_entries):
            default_name = entry.get("name", f"Target{i + 1}")
            if custom_labels and i < len(custom_labels) and custom_labels[i]:
                label = custom_labels[i]
            else:
                label = default_name
            dropdown.addItem(label, i)

        dropdown.blockSignals(False)
        self.update_target_controls()

    # --- exported helpers --------------------------------------------------

    def update_trajectory_colorbar(
        self,
        color_mode: str,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        """Show or hide the trajectory colorbar and update its labels.

        Args:
            color_mode: Active trajectory color mode.
            vmin: Minimum scalar value (shown on the left).
            vmax: Maximum scalar value (shown on the right).
        """
        container = self.widgets.get("trajectory_colorbar_container")
        if container is None:
            return

        if color_mode == "node_color":
            container.setVisible(False)
            return

        container.setVisible(True)
        units = {"speed": "m/frame", "altitude": "m", "time": "frame", "angular_speed": "rad/frame"}
        unit = units.get(color_mode, "")

        min_label = self.widgets.get("trajectory_colorbar_min_label")
        max_label = self.widgets.get("trajectory_colorbar_max_label")
        if min_label and vmin is not None:
            min_label.setText(f"{vmin:.1f}")
            min_label.setToolTip(f"Min {color_mode}: {vmin:.2f} {unit}")
        elif min_label:
            min_label.setText("")
        if max_label and vmax is not None:
            max_label.setText(f"{vmax:.1f}")
            max_label.setToolTip(f"Max {color_mode}: {vmax:.2f} {unit}")
        elif max_label:
            max_label.setText("")
