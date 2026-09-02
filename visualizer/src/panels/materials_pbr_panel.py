"""Material appearance controls shared by Open3D and pygfx renderers.

The panel edits visual material state, not RF material semantics. Material IDs,
visibility modes, PBR overrides, visual profiles, and texture edit locks are
applied through the material services and renderer capability map.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from shared.logging import get_logger

from ..materials.appearance import MaterialDisplayMode, VisualMaterialSource
from ..renderers.protocol import renderer_capabilities
from ..state import (
    DEFAULT_RF_XRAY_OPACITY,
    DEFAULT_RF_XRAY_PROPERTY,
    RF_XRAY_PROPERTY_OPTIONS,
    normalize_rf_xray_opacity,
)
from .base import BasePanel
from .ui_theme import configure_label, set_widget_role

if TYPE_CHECKING:
    from ..pipeline.core import Visualizer

logger = get_logger("orchav.materials_panel")

_RF_XRAY_PROPERTY_LABELS = {
    "relative_permittivity": "Relative Permittivity",
    "conductivity": "Conductivity",
    "scattering_coefficient": "Scattering Coefficient",
    "xpd_coefficient": "XPD Coefficient",
    "thickness": "Thickness",
}


class MaterialsPanel(BasePanel):
    """Build the Materials panel for visibility, color, PBR, and presets."""

    def __init__(self, parent_widget: "Visualizer"):
        """Initialize signal guards and texture-driven color-edit state."""
        super().__init__(parent_widget)
        self._suppress_signals = False
        self._color_editing_texture_locked = False

    def create_panel(self):
        """Create and return the materials PBR panel."""
        group = self.create_group_box("Materials")

        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        # Material type selector
        material_layout = QHBoxLayout()
        material_layout.addWidget(QLabel("Visual Material:"))
        self.widgets["material_combo"] = QComboBox()
        self.widgets["material_combo"].setToolTip(
            "Select the material type whose visual appearance should be adjusted"
        )
        self.widgets["material_combo"].currentTextChanged.connect(self._on_material_changed)
        material_layout.addWidget(self.widgets["material_combo"], 1)
        layout.addLayout(material_layout)

        # EM / Visual profile info section
        profile_group = self.create_subgroup_box("Material Identity")
        profile_layout = QVBoxLayout(profile_group)
        profile_layout.setSpacing(2)

        em_row = QHBoxLayout()
        em_row.addWidget(QLabel("EM Material:"))
        self.widgets["em_material_label"] = QLabel("--")
        self.widgets["em_material_label"].setToolTip(
            "Electromagnetic material used by Sionna RT for simulation"
        )
        self.widgets["em_material_label"].setStyleSheet("font-weight: bold;")
        em_row.addWidget(self.widgets["em_material_label"], 1)
        profile_layout.addLayout(em_row)

        vis_row = QHBoxLayout()
        vis_row.addWidget(QLabel("Visual Preset:"))
        self.widgets["visual_preset_label"] = QLabel("--")
        self.widgets["visual_preset_label"].setToolTip(
            "PBR preset controlling visual appearance (from profile or ITU default)"
        )
        self.widgets["visual_preset_label"].setStyleSheet("font-weight: bold;")
        vis_row.addWidget(self.widgets["visual_preset_label"], 1)

        self.widgets["clear_profile_btn"] = QPushButton("Clear")
        self.widgets["clear_profile_btn"].setFixedWidth(50)
        self.widgets["clear_profile_btn"].setToolTip(
            "Remove the visual assignment and follow the current EM material"
        )
        self.widgets["clear_profile_btn"].clicked.connect(self._on_clear_profile)
        self.widgets["clear_profile_btn"].setVisible(False)
        vis_row.addWidget(self.widgets["clear_profile_btn"])
        profile_layout.addLayout(vis_row)

        layout.addWidget(profile_group)
        self.widgets["profile_group"] = profile_group

        # Color picker row
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Appearance Color:"))
        self.widgets["color_btn"] = QPushButton()
        self.widgets["color_btn"].setFixedSize(60, 25)
        self.widgets["color_btn"].setToolTip("Click to change this material type's visual color")
        self.widgets["color_btn"].clicked.connect(self._on_color_clicked)
        self._current_color = [0.7, 0.7, 0.7]  # Default gray
        self._update_color_button()
        color_layout.addWidget(self.widgets["color_btn"])
        color_layout.addStretch()
        layout.addLayout(color_layout)

        # Transient material overlays. Neither toggle selected means normal.
        visibility_layout = QHBoxLayout()
        visibility_layout.addWidget(QLabel("Display:"))

        # Hide button (red X)
        self.widgets["hide_btn"] = QPushButton("Hide")
        self.widgets["hide_btn"].setCheckable(True)
        self.widgets["hide_btn"].setToolTip(
            "Hide this material; click again to restore its normal appearance"
        )
        self.widgets["hide_btn"].setStyleSheet("""
            QPushButton { background-color: #f44336; color: white; border-radius: 3px; padding: 4px 8px; }
            QPushButton:checked { background-color: #c62828; border: 2px solid #b71c1c; }
            QPushButton:hover { background-color: #ef5350; }
            """)
        self.widgets["hide_btn"].clicked.connect(
            lambda: self._on_visibility_changed(MaterialDisplayMode.HIDDEN)
        )
        visibility_layout.addWidget(self.widgets["hide_btn"])

        # Highlight button (yellow star)
        self.widgets["highlight_btn"] = QPushButton("Highlight")
        self.widgets["highlight_btn"].setCheckable(True)
        self.widgets["highlight_btn"].setToolTip(
            "Highlight this material; click again to restore its normal appearance"
        )
        self.widgets["highlight_btn"].setStyleSheet("""
            QPushButton { background-color: #ffc107; color: #333; border-radius: 3px; padding: 4px 8px; }
            QPushButton:checked { background-color: #ff8f00; border: 2px solid #e65100; }
            QPushButton:hover { background-color: #ffca28; }
            """)
        self.widgets["highlight_btn"].clicked.connect(
            lambda: self._on_visibility_changed(MaterialDisplayMode.HIGHLIGHTED)
        )
        visibility_layout.addWidget(self.widgets["highlight_btn"])

        visibility_layout.addStretch()
        layout.addLayout(visibility_layout)

        # Visual Properties section (PBR-capable renderers)
        self.widgets["pbr_group"] = self.create_subgroup_box("PBR Properties")
        visual_group = self.widgets["pbr_group"]
        visual_layout = QVBoxLayout(visual_group)
        visual_layout.setSpacing(4)

        visual_layout.addLayout(
            self._create_property_grid(
                [
                    ("roughness", 0.0, 1.0, 0.01, "Roughness (0=mirror, 1=matte)"),
                    ("metallic", 0.0, 1.0, 0.01, "Metallic (0=plastic, 1=metal)"),
                    ("reflectance", 0.0, 1.0, 0.01, "Reflectance (light reflection)"),
                    ("alpha", 0.0, 1.0, 0.05, "Alpha (0=transparent, 1=opaque)"),
                ]
            )
        )

        layout.addWidget(visual_group)

        # Advanced PBR section — collapsible, default-collapsed. Explicit
        # material-feature capabilities disable unsupported controls with a
        # reason while preserving their semantic values for renderer changes.
        # Note: glass_thickness is intentionally distinct from any
        # XML-derived wall/object thickness metadata.
        self.widgets["advanced_pbr_group"] = self.create_subgroup_box("Advanced PBR")
        advanced_group = self.widgets["advanced_pbr_group"]
        advanced_group.setCheckable(True)
        advanced_group.setChecked(False)
        advanced_layout = QVBoxLayout(advanced_group)
        advanced_layout.setSpacing(4)

        advanced_content = QWidget()
        self.widgets["advanced_pbr_content"] = advanced_content
        advanced_content_layout = QVBoxLayout(advanced_content)
        advanced_content_layout.setContentsMargins(0, 0, 0, 0)
        advanced_content_layout.setSpacing(4)

        wave_b_tooltip = "Open3D renderer only — pygfx silently discards this."
        advanced_content_layout.addLayout(
            self._create_property_grid(
                [
                    ("clearcoat", 0.0, 1.0, 0.01, "Clearcoat layer strength"),
                    (
                        "clearcoat_roughness",
                        0.0,
                        1.0,
                        0.01,
                        "Roughness of the clearcoat layer",
                    ),
                    (
                        "anisotropy",
                        0.0,
                        1.0,
                        0.01,
                        "Directional highlight strength (brushed metal)",
                    ),
                    (
                        "emissive_intensity",
                        0.0,
                        5.0,
                        0.05,
                        "Self-illumination intensity multiplier (uses material color by default)",
                    ),
                    (
                        "normal_map_strength",
                        0.0,
                        3.0,
                        0.05,
                        "Normal-map relief strength (shading only; requires a normal map)",
                    ),
                    (
                        "transmission",
                        0.0,
                        1.0,
                        0.01,
                        f"Glass transmission. {wave_b_tooltip}",
                    ),
                    (
                        "glass_thickness",
                        0.0,
                        5.0,
                        0.05,
                        f"Volumetric glass thickness. {wave_b_tooltip}",
                    ),
                ]
            )
        )

        advanced_layout.addWidget(advanced_content)
        advanced_group.toggled.connect(self._set_advanced_pbr_expanded)
        self._set_advanced_pbr_expanded(False)

        layout.addWidget(advanced_group)

        # Action buttons
        button_row = QHBoxLayout()
        self.widgets["reset_btn"] = QPushButton("Reset to Default")
        self.widgets["reset_btn"].setToolTip(
            "Remove runtime edits and restore the scenario profile or EM-derived appearance"
        )
        self.widgets["reset_btn"].clicked.connect(self._on_reset_material)
        button_row.addWidget(self.widgets["reset_btn"])

        self.widgets["reset_all_btn"] = QPushButton("Reset All")
        self.widgets["reset_all_btn"].setToolTip(
            "Remove all runtime material edits and restore underlying assignments"
        )
        self.widgets["reset_all_btn"].clicked.connect(self._on_reset_all)
        button_row.addWidget(self.widgets["reset_all_btn"])
        layout.addLayout(button_row)

        # Preset section
        preset_group = self.create_subgroup_box("Presets")
        preset_layout = QVBoxLayout(preset_group)

        preset_combo_row = QHBoxLayout()
        preset_combo_row.setSpacing(4)
        preset_combo_row.addWidget(QLabel("Built-in:"))
        self.widgets["preset_combo"] = QComboBox()
        self.widgets["preset_combo"].setToolTip("Apply a built-in material preset")
        self.widgets["preset_combo"].currentIndexChanged.connect(self._on_preset_selected)
        preset_combo_row.addWidget(self.widgets["preset_combo"], 1)
        self.widgets["apply_preset_btn"] = QPushButton("Apply")
        self.widgets["apply_preset_btn"].setToolTip("Apply selected preset")
        self.widgets["apply_preset_btn"].clicked.connect(self._on_apply_preset)
        preset_combo_row.addWidget(self.widgets["apply_preset_btn"])
        self.widgets["save_preset_btn"] = QPushButton("Save")
        self.widgets["save_preset_btn"].setToolTip("Save current settings as custom preset")
        self.widgets["save_preset_btn"].clicked.connect(self._on_save_preset)
        preset_combo_row.addWidget(self.widgets["save_preset_btn"])

        self.widgets["load_preset_btn"] = QPushButton("Load")
        self.widgets["load_preset_btn"].setToolTip("Load a custom preset")
        self.widgets["load_preset_btn"].clicked.connect(self._on_load_preset)
        preset_combo_row.addWidget(self.widgets["load_preset_btn"])
        preset_layout.addLayout(preset_combo_row)

        layout.addWidget(preset_group)

        layout.addStretch()

        self._refresh_preset_combo()

        return group

    def create_rf_xray_panel(self):
        """Create RF analysis controls for their capability-gated section."""
        return self._create_rf_xray_group()

    def _create_rf_xray_group(self):
        """Create RF X-Ray overlay controls for material inspection."""
        group = self.create_subgroup_box("RF X-Ray")
        rf_layout = QVBoxLayout(group)
        rf_layout.setSpacing(4)

        toggle_row = QHBoxLayout()
        self.widgets["rf_xray_toggle"] = QCheckBox("Enable")
        self.widgets["rf_xray_toggle"].setToolTip(
            "Inspect scene material assignments or current MPC material contribution"
        )
        self.widgets["rf_xray_toggle"].toggled.connect(self._on_rf_xray_toggled)
        toggle_row.addWidget(self.widgets["rf_xray_toggle"])

        self.widgets["rf_xray_mode_combo"] = QComboBox()
        self.widgets["rf_xray_mode_combo"].addItem("Material Map", "material_map")
        self.widgets["rf_xray_mode_combo"].addItem("MPC Material Usage", "mpc_usage")
        self.widgets["rf_xray_mode_combo"].addItem("Material Properties", "material_properties")
        self.widgets["rf_xray_mode_combo"].setToolTip(
            "Material Map recolors assigned materials; MPC Material Usage uses path-loss-weighted material contacts; Material Properties colors configured RF scalars"
        )
        self.widgets["rf_xray_mode_combo"].currentIndexChanged.connect(
            self._on_rf_xray_mode_changed
        )
        toggle_row.addWidget(self.widgets["rf_xray_mode_combo"], 1)
        rf_layout.addLayout(toggle_row)

        property_row = QHBoxLayout()
        self.widgets["rf_xray_property_label"] = QLabel("Property:")
        property_row.addWidget(self.widgets["rf_xray_property_label"])
        self.widgets["rf_xray_property_combo"] = QComboBox()
        for prop in RF_XRAY_PROPERTY_OPTIONS:
            self.widgets["rf_xray_property_combo"].addItem(
                _RF_XRAY_PROPERTY_LABELS.get(prop, prop),
                prop,
            )
        self.widgets["rf_xray_property_combo"].setToolTip(
            "Select the configured RF material property for Material Properties mode"
        )
        self.widgets["rf_xray_property_combo"].currentIndexChanged.connect(
            self._on_rf_xray_property_changed
        )
        property_row.addWidget(self.widgets["rf_xray_property_combo"], 1)
        rf_layout.addLayout(property_row)

        opacity_row = QHBoxLayout()
        self.widgets["rf_xray_opacity_label"] = QLabel("")
        opacity_row.addWidget(self.widgets["rf_xray_opacity_label"])
        self.widgets["rf_xray_opacity_slider"] = QSlider(Qt.Horizontal)
        self.widgets["rf_xray_opacity_slider"].setRange(5, 100)
        self.widgets["rf_xray_opacity_slider"].setSingleStep(5)
        self.widgets["rf_xray_opacity_slider"].setPageStep(10)
        self.widgets["rf_xray_opacity_slider"].setToolTip(
            "Adjust RF X-Ray material overlay opacity"
        )
        self.widgets["rf_xray_opacity_slider"].valueChanged.connect(
            self._on_rf_xray_opacity_changed
        )
        opacity_row.addWidget(self.widgets["rf_xray_opacity_slider"], 1)
        rf_layout.addLayout(opacity_row)

        marker_row = QHBoxLayout()
        self.widgets["rf_xray_top_paths_cb"] = QCheckBox("Top Paths")
        self.widgets["rf_xray_top_paths_cb"].setToolTip(
            "Highlight the strongest material-bearing paths"
        )
        self.widgets["rf_xray_top_paths_cb"].toggled.connect(self._on_rf_xray_top_paths_toggled)
        marker_row.addWidget(self.widgets["rf_xray_top_paths_cb"])

        marker_row.addWidget(QLabel("Max:"))
        self.widgets["rf_xray_top_paths_spin"] = QSpinBox()
        self.widgets["rf_xray_top_paths_spin"].setRange(1, 100)
        self.widgets["rf_xray_top_paths_spin"].setValue(12)
        self.widgets["rf_xray_top_paths_spin"].setToolTip("Maximum strongest paths to highlight")
        self.widgets["rf_xray_top_paths_spin"].valueChanged.connect(
            self._on_rf_xray_max_paths_changed
        )
        marker_row.addWidget(self.widgets["rf_xray_top_paths_spin"])
        rf_layout.addLayout(marker_row)

        self.widgets["rf_xray_status"] = QLabel("")
        self.widgets["rf_xray_status"].setWordWrap(True)
        configure_label(
            self.widgets["rf_xray_status"],
            role="muted",
            italic=True,
        )
        rf_layout.addWidget(self.widgets["rf_xray_status"])

        self._sync_rf_xray_controls()
        return group

    def _rf_xray_supported(self) -> bool:
        """Return whether the current renderer can present RF X-Ray overlays."""
        return renderer_capabilities(getattr(self.parent, "renderer", None)).rf_xray_overlay

    def _sync_rf_xray_controls(self) -> None:
        """Mirror AppState and renderer support into RF X-Ray controls."""
        state = getattr(self.parent, "app_state", None)
        supported = self._rf_xray_supported()
        self._suppress_signals = True
        try:
            toggle = self.widgets.get("rf_xray_toggle")
            if toggle is not None:
                toggle.setChecked(supported and bool(getattr(state, "show_rf_xray", False)))
            combo = self.widgets.get("rf_xray_mode_combo")
            if combo is not None:
                mode = str(getattr(state, "rf_xray_mode", "material_map"))
                index = combo.findData(mode)
                combo.setCurrentIndex(index if index >= 0 else 0)
            property_combo = self.widgets.get("rf_xray_property_combo")
            if property_combo is not None:
                prop = str(getattr(state, "rf_xray_property", DEFAULT_RF_XRAY_PROPERTY))
                index = property_combo.findData(prop)
                property_combo.setCurrentIndex(index if index >= 0 else 0)
            opacity = normalize_rf_xray_opacity(
                getattr(state, "rf_xray_opacity", DEFAULT_RF_XRAY_OPACITY)
            )
            opacity_value = int(round(opacity * 100.0))
            opacity_slider = self.widgets.get("rf_xray_opacity_slider")
            if opacity_slider is not None:
                opacity_slider.setValue(opacity_value)
            self._set_rf_xray_opacity_label(opacity_value)
            top_paths = self.widgets.get("rf_xray_top_paths_cb")
            if top_paths is not None:
                top_paths.setChecked(bool(getattr(state, "rf_xray_show_top_paths", False)))
            max_paths = self.widgets.get("rf_xray_top_paths_spin")
            if max_paths is not None:
                max_paths.setValue(int(getattr(state, "rf_xray_max_top_paths", 12)))
        finally:
            self._suppress_signals = False

        for key in (
            "rf_xray_toggle",
            "rf_xray_mode_combo",
            "rf_xray_property_label",
            "rf_xray_property_combo",
            "rf_xray_opacity_label",
            "rf_xray_opacity_slider",
            "rf_xray_top_paths_cb",
            "rf_xray_top_paths_spin",
        ):
            widget = self.widgets.get(key)
            if widget is not None:
                widget.setEnabled(supported)

        property_mode = str(getattr(state, "rf_xray_mode", "material_map")) == (
            "material_properties"
        )
        for key in ("rf_xray_property_label", "rf_xray_property_combo"):
            widget = self.widgets.get(key)
            if widget is not None:
                widget.setEnabled(supported and property_mode)

        if not supported:
            self.set_rf_xray_status("RF X-Ray overlay is pygfx-only.", active=False)
        else:
            self.set_rf_xray_status("RF X-Ray ready.", active=False)

    def set_rf_xray_status(self, text: str, *, active: bool) -> None:
        """Update the RF X-Ray status line."""
        label = self.widgets.get("rf_xray_status")
        if label is None:
            return
        label.setText(str(text or ""))
        font = label.font()
        font.setBold(bool(active))
        font.setItalic(not active)
        label.setFont(font)
        set_widget_role(label, "accent" if active else "muted")

    def _on_rf_xray_toggled(self, checked: bool) -> None:
        """Forward RF X-Ray visibility changes to the controller."""
        if self._suppress_signals:
            return
        controller = getattr(self.parent, "ui_controller", None)
        if controller is not None and hasattr(controller, "handle_rf_xray_toggled"):
            controller.handle_rf_xray_toggled(bool(checked))

    def _on_rf_xray_mode_changed(self, _index: int) -> None:
        """Forward RF X-Ray mode changes to the controller."""
        if self._suppress_signals:
            return
        combo = self.widgets.get("rf_xray_mode_combo")
        if combo is None:
            return
        mode = combo.currentData() or "material_map"
        controller = getattr(self.parent, "ui_controller", None)
        if controller is not None and hasattr(controller, "handle_rf_xray_mode_changed"):
            controller.handle_rf_xray_mode_changed(str(mode))

    def _on_rf_xray_property_changed(self, _index: int) -> None:
        """Forward RF X-Ray material-property changes to the controller."""
        if self._suppress_signals:
            return
        combo = self.widgets.get("rf_xray_property_combo")
        if combo is None:
            return
        prop = combo.currentData() or DEFAULT_RF_XRAY_PROPERTY
        controller = getattr(self.parent, "ui_controller", None)
        if controller is not None and hasattr(controller, "handle_rf_xray_property_changed"):
            controller.handle_rf_xray_property_changed(str(prop))

    def _set_rf_xray_opacity_label(self, value: int) -> None:
        """Render the RF X-Ray opacity percentage beside the slider."""
        label = self.widgets.get("rf_xray_opacity_label")
        if label is not None:
            label.setText(f"Opacity: {int(value)}%")

    def _on_rf_xray_opacity_changed(self, value: int) -> None:
        """Forward RF X-Ray material overlay opacity changes."""
        self._set_rf_xray_opacity_label(int(value))
        if self._suppress_signals:
            return
        controller = getattr(self.parent, "ui_controller", None)
        if controller is not None and hasattr(controller, "handle_rf_xray_opacity_changed"):
            controller.handle_rf_xray_opacity_changed(int(value))

    def _on_rf_xray_top_paths_toggled(self, checked: bool) -> None:
        """Forward RF X-Ray top-path visibility changes."""
        if self._suppress_signals:
            return
        controller = getattr(self.parent, "ui_controller", None)
        if controller is not None and hasattr(controller, "handle_rf_xray_top_paths_toggled"):
            controller.handle_rf_xray_top_paths_toggled(bool(checked))

    def _on_rf_xray_max_paths_changed(self, value: int) -> None:
        """Forward RF X-Ray strongest-path cap changes."""
        if self._suppress_signals:
            return
        controller = getattr(self.parent, "ui_controller", None)
        if controller is not None and hasattr(controller, "handle_rf_xray_max_paths_changed"):
            controller.handle_rf_xray_max_paths_changed(int(value))

    def _set_advanced_pbr_expanded(self, expanded: bool) -> None:
        """Show Advanced PBR controls only when the section is expanded."""
        content = self.widgets.get("advanced_pbr_content")
        if content is not None:
            content.setVisible(bool(expanded))

    def _create_property_slider(
        self, prop_name: str, min_val: float, max_val: float, step: float, tooltip: str
    ) -> QHBoxLayout:
        """Create a property control with label, slider, and spinbox.

        Args:
            prop_name: Property name (e.g., "roughness")
            min_val: Minimum value
            max_val: Maximum value
            step: Step size
            tooltip: Tooltip text

        Returns:
            QHBoxLayout with the controls
        """
        row = QHBoxLayout()

        # Label
        label_text = prop_name.capitalize() + ":"
        label = QLabel(label_text)
        label.setMinimumWidth(80)
        row.addWidget(label)

        # Slider
        slider_name = f"{prop_name}_slider"
        self.widgets[slider_name] = QSlider(Qt.Horizontal)
        # Map float range to int range (0-1000)
        self.widgets[slider_name].setRange(0, 1000)
        self.widgets[slider_name].setValue(500)
        self.widgets[slider_name].setToolTip(tooltip)
        self.widgets[slider_name].valueChanged.connect(
            lambda v, p=prop_name: self._on_slider_changed(p, v)
        )
        row.addWidget(self.widgets[slider_name], 1)

        # SpinBox
        spin_name = f"{prop_name}_spin"
        self.widgets[spin_name] = QDoubleSpinBox()
        self.widgets[spin_name].setRange(min_val, max_val)
        self.widgets[spin_name].setSingleStep(step)
        self.widgets[spin_name].setValue((min_val + max_val) / 2)
        self.widgets[spin_name].setDecimals(2)
        self.widgets[spin_name].setToolTip(tooltip)
        self.widgets[spin_name].valueChanged.connect(
            lambda v, p=prop_name: self._on_spin_changed(p, v)
        )
        row.addWidget(self.widgets[spin_name])

        return row

    def _create_property_grid(self, properties: list[tuple[str, float, float, float, str]]):
        """Create a compact two-column grid of property slider rows."""
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        for index, prop in enumerate(properties):
            row = index // 2
            column = index % 2
            grid.addLayout(self._create_property_slider(*prop), row, column)
        return grid

    def _on_slider_changed(self, prop_name: str, slider_value: int):
        """Handle slider value change."""
        if self._suppress_signals:
            return

        # Map slider (0-1000) to property range
        spin_widget = self.widgets.get(f"{prop_name}_spin")
        if spin_widget is None:
            return

        min_val = spin_widget.minimum()
        max_val = spin_widget.maximum()
        prop_value = min_val + (max_val - min_val) * (slider_value / 1000.0)

        self._suppress_signals = True
        spin_widget.setValue(prop_value)
        self._suppress_signals = False

        self._apply_property_change(prop_name, prop_value)

    def _on_spin_changed(self, prop_name: str, prop_value: float):
        """Handle spinbox value change."""
        if self._suppress_signals:
            return

        spin_widget = self.widgets.get(f"{prop_name}_spin")
        slider_widget = self.widgets.get(f"{prop_name}_slider")
        if spin_widget is None or slider_widget is None:
            return

        min_val = spin_widget.minimum()
        max_val = spin_widget.maximum()
        slider_value = int(1000.0 * (prop_value - min_val) / (max_val - min_val))

        self._suppress_signals = True
        slider_widget.setValue(slider_value)
        self._suppress_signals = False

        self._apply_property_change(prop_name, prop_value)

    def _set_property_ui(self, prop_name: str, value: float):
        """Set both slider and spinbox to a value without triggering signals.

        Args:
            prop_name: Property name (e.g., "roughness")
            value: Value to set
        """
        spin_widget = self.widgets.get(f"{prop_name}_spin")
        slider_widget = self.widgets.get(f"{prop_name}_slider")

        if spin_widget is None:
            return

        spin_widget.setValue(value)

        if slider_widget is not None:
            min_val = spin_widget.minimum()
            max_val = spin_widget.maximum()
            if max_val > min_val:
                slider_value = int(1000.0 * (value - min_val) / (max_val - min_val))
                slider_widget.setValue(slider_value)

    def _apply_property_change(self, prop_name: str, value: float):
        """Apply a property change to the selected material."""
        material_combo = self.widgets.get("material_combo")
        if material_combo is None:
            return

        material_type = material_combo.currentText()
        if not material_type:
            return

        if hasattr(self.parent, "material_pbr_service"):
            self.parent.material_pbr_service.set_property(material_type, prop_name, value)
            # Mark as manually overridden in the visual preset label.
            # Guard against re-appending: the suffix is " + override" so check
            # for that exact marker, not a different string.
            vis_label = self.widgets.get("visual_preset_label")
            if vis_label is not None and "+ override" not in (vis_label.text() or ""):
                vis_label.setText(f"{vis_label.text()} + override")
                vis_label.setStyleSheet("font-weight: bold; color: #FF9800;")
            logger.debug(f"Applied {material_type}.{prop_name} = {value:.2f}")

    def _update_color_button(self):
        """Update the color button's background to show current color."""
        color_btn = self.widgets.get("color_btn")
        if color_btn is None:
            return

        r = int(self._current_color[0] * 255)
        g = int(self._current_color[1] * 255)
        b = int(self._current_color[2] * 255)
        color_btn.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border: 1px solid #555;")
        self._apply_color_button_enabled_state()

    def _set_color_editing_texture_locked(self, locked: bool):
        """Disable color edits when no group member has uniform color ownership."""
        self._color_editing_texture_locked = bool(locked)
        self._apply_color_button_enabled_state()

    def _apply_color_button_enabled_state(self):
        """Apply the current texture-policy lock to the color picker button."""
        color_btn = self.widgets.get("color_btn")
        if color_btn is None:
            return
        locked = bool(getattr(self, "_color_editing_texture_locked", False))
        color_btn.setEnabled(not locked)
        if locked:
            color_btn.setToolTip(
                getattr(
                    self,
                    "_color_edit_note",
                    "Authored textures or vertex colors own this group's RGB.",
                )
            )
        else:
            color_btn.setToolTip(
                getattr(
                    self,
                    "_color_edit_note",
                    "Click to change this material type's visual color",
                )
            )

    def _on_color_clicked(self):
        """Handle color button click - open color picker dialog."""
        if self._color_editing_texture_locked:
            return
        r = int(self._current_color[0] * 255)
        g = int(self._current_color[1] * 255)
        b = int(self._current_color[2] * 255)
        current_qcolor = QColor(r, g, b)

        # Open color dialog
        color = QColorDialog.getColor(current_qcolor, self.parent, "Select Material Color")

        if color.isValid():
            self._current_color = [
                color.redF(),
                color.greenF(),
                color.blueF(),
            ]
            self._update_color_button()

            material_combo = self.widgets.get("material_combo")
            if material_combo is not None:
                material_type = material_combo.currentText()
                if material_type and hasattr(self.parent, "material_pbr_service"):
                    self.parent.material_pbr_service.set_property(
                        material_type, "color", self._current_color
                    )
                    logger.info(
                        f"Applied color [{self._current_color[0]:.2f}, "
                        f"{self._current_color[1]:.2f}, {self._current_color[2]:.2f}] "
                        f"to '{material_type}'"
                    )

    def _on_visibility_changed(self, mode: MaterialDisplayMode | str):
        """Handle visibility button click for the selected material."""
        material_combo = self.widgets.get("material_combo")
        if material_combo is None:
            return

        material_type = material_combo.currentText()
        if not material_type:
            return

        material_mode_service = getattr(self.parent, "material_mode_service", None)
        if material_mode_service is None:
            logger.warning("MaterialModeService not available for visibility change")
            return
        next_mode = material_mode_service.resolve_toggled_mode(material_type, mode)
        material_mode_service.set_mode(material_type, next_mode)

        hide_btn = self.widgets.get("hide_btn")
        highlight_btn = self.widgets.get("highlight_btn")

        if hide_btn is not None:
            hide_btn.setChecked(next_mode is MaterialDisplayMode.HIDDEN)
        if highlight_btn is not None:
            highlight_btn.setChecked(next_mode is MaterialDisplayMode.HIGHLIGHTED)

        ui_controller = getattr(self.parent, "ui_controller", None)
        if ui_controller is not None:
            ui_controller.apply_material_modes(material_type)

        logger.info("Set visibility for '%s' to '%s'", material_type, next_mode)

    def _update_visibility_buttons(self, material_type: str):
        """Update visibility button states for the given material."""
        material_mode_service = getattr(self.parent, "material_mode_service", None)
        if material_mode_service is None:
            return
        mode = MaterialDisplayMode.coerce(material_mode_service.get_mode(material_type))

        hide_btn = self.widgets.get("hide_btn")
        highlight_btn = self.widgets.get("highlight_btn")

        if hide_btn is not None:
            hide_btn.setChecked(mode is MaterialDisplayMode.HIDDEN)
        if highlight_btn is not None:
            highlight_btn.setChecked(mode is MaterialDisplayMode.HIGHLIGHTED)

    def _on_material_changed(self, material_type: str):
        """Handle material selection change."""
        if not material_type:
            return

        self._load_material_properties(material_type)

        self._update_visibility_buttons(material_type)

        self._update_profile_info(material_type)

    def _load_material_properties(self, material_type: str):
        """Load and display properties for a material type."""
        if not hasattr(self.parent, "mesh_entries"):
            return

        pbr_service = getattr(self.parent, "material_pbr_service", None)
        summary = (
            pbr_service.summarize_material_group(material_type) if pbr_service is not None else None
        )
        if summary is None or summary.member_count == 0:
            logger.warning(f"No properties found for material '{material_type}'")
            return
        props = dict(summary.representative_properties)

        # Core PBR fields plus the advanced material controls.
        # Most advanced sliders reset to 0.0 when the material has no override.
        # ``normal_map_strength`` resets to 1.0 because that is the authored /
        # neutral normal-map scale.
        self._suppress_signals = True
        legacy_fields = ["roughness", "metallic", "reflectance", "alpha"]
        advanced_fields = [
            "clearcoat",
            "clearcoat_roughness",
            "anisotropy",
            "emissive_intensity",
            "normal_map_strength",
            "transmission",
            "glass_thickness",
        ]
        advanced_defaults = {
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0,
            "emissive_intensity": 0.0,
            "normal_map_strength": 1.0,
            "transmission": 0.0,
            "glass_thickness": 0.0,
        }
        for prop_name in legacy_fields + advanced_fields:
            if prop_name in props:
                self._set_property_ui(prop_name, props[prop_name])
            elif prop_name in advanced_fields:
                # Material lacks this advanced field — reset to neutral.
                self._set_property_ui(prop_name, advanced_defaults[prop_name])

        if "color" in props:
            color = props["color"]
            if isinstance(color, (list, tuple)) and len(color) >= 3:
                self._current_color = [color[0], color[1], color[2]]
                self._update_color_button()
        self._color_edit_note = summary.color_edit_note
        self._set_color_editing_texture_locked(not summary.color_editable)

        self._suppress_signals = False

    def _update_profile_info(self, material_type: str) -> None:
        """Update the EM Material / Visual Preset labels for the selected material."""
        em_label = self.widgets.get("em_material_label")
        vis_label = self.widgets.get("visual_preset_label")
        clear_btn = self.widgets.get("clear_profile_btn")

        if em_label is None or vis_label is None:
            return

        pbr_service = getattr(self.parent, "material_pbr_service", None)
        get_visual_key = getattr(pbr_service, "get_visual_material_key", None)
        get_binding = getattr(pbr_service, "get_visual_binding", None)
        matching_entries = []
        bindings = []
        if callable(get_visual_key) and callable(get_binding):
            all_entries = list(getattr(self.parent, "mesh_entries", []))
            all_entries += list(getattr(self.parent, "target_entries", []))
            matching_entries = [
                entry for entry in all_entries if str(get_visual_key(entry)) == str(material_type)
            ]
            bindings = [get_binding(entry) for entry in matching_entries]

        em_materials = sorted(
            {str(entry.get("material_type") or "--") for entry in matching_entries}
        )
        em_label.setToolTip("Electromagnetic material used by Sionna RT for simulation")
        if not em_materials:
            em_label.setText(str(material_type or "--"))
        elif len(em_materials) == 1:
            em_label.setText(em_materials[0])
        else:
            em_label.setText("Mixed")
            em_label.setToolTip("EM materials: " + ", ".join(em_materials))

        non_follow_bindings = [
            binding
            for binding in bindings
            if getattr(binding, "source", VisualMaterialSource.FOLLOW_EM)
            is not VisualMaterialSource.FOLLOW_EM
        ]
        if not non_follow_bindings:
            vis_label.setText("Follow EM")
            vis_label.setStyleSheet("font-weight: bold;")
            if clear_btn is not None:
                clear_btn.setVisible(False)
            return

        assignment_signatures = {
            (
                getattr(binding, "source", VisualMaterialSource.FOLLOW_EM),
                getattr(binding, "preset", None),
                getattr(binding, "material_type", None),
            )
            for binding in non_follow_bindings
        }
        if len(assignment_signatures) == 1:
            source, preset, bound_material = next(iter(assignment_signatures))
            assignment_name = preset or bound_material or material_type
            source_tag = "manual" if source is VisualMaterialSource.MANUAL else "profile"
            suffix = " + Follow EM" if len(non_follow_bindings) != len(bindings) else ""
            vis_label.setText(f"{assignment_name} ({source_tag}){suffix}")
            color = "#4CAF50" if source is VisualMaterialSource.MANUAL else "#2196F3"
            vis_label.setStyleSheet(f"font-weight: bold; color: {color};")
        else:
            vis_label.setText("Mixed visual assignments")
            vis_label.setStyleSheet("font-weight: bold; color: #FF9800;")
        if clear_btn is not None:
            clear_btn.setVisible(True)

    def _on_clear_profile(self) -> None:
        """Clear the current visual assignment and return the group to Follow EM."""
        material_combo = self.widgets.get("material_combo")
        if material_combo is None:
            return

        material_type = material_combo.currentText()
        if not material_type:
            return

        pbr_service = getattr(self.parent, "material_pbr_service", None)
        clear_assignment = getattr(pbr_service, "clear_visual_assignment", None)
        if not callable(clear_assignment) or not clear_assignment(material_type):
            return

        self.refresh_material_list()
        logger.info("Cleared visual assignment for '%s', reverted to Follow EM", material_type)

    def _on_reset_material(self):
        """Reset current material to default values."""
        material_combo = self.widgets.get("material_combo")
        if material_combo is None:
            return

        material_type = material_combo.currentText()
        if not material_type:
            return

        if hasattr(self.parent, "material_pbr_service"):
            if self.parent.material_pbr_service.reset_material(material_type):
                self.refresh_material_list()
                logger.info("Reset material '%s' to its underlying assignment", material_type)

    def _on_reset_all(self):
        """Reset all materials to default values."""
        if hasattr(self.parent, "material_pbr_service"):
            if self.parent.material_pbr_service.reset_all():
                self.refresh_material_list()
                logger.info("Reset all manual material overrides")

    def _refresh_preset_combo(self):
        """Refresh the preset combo box with built-in presets.

        Presets are material-agnostic and can be applied to any selected material.
        """
        preset_combo = self.widgets.get("preset_combo")
        if preset_combo is None:
            return

        preset_combo.clear()
        if hasattr(self.parent, "material_pbr_service"):
            from ..materials.presets import BUILTIN_MATERIAL_PRESETS

            for preset_name, preset_data in BUILTIN_MATERIAL_PRESETS.items():
                description = preset_data.get("description", "")
                preset_combo.addItem(preset_name, preset_name)
                preset_combo.setItemData(preset_combo.count() - 1, description, Qt.ToolTipRole)

        # General tooltip for the combo
        preset_combo.setToolTip(
            "Select a preset to preview values.\n"
            "Click 'Apply' to apply the preset to the selected material."
        )

    def _on_preset_selected(self, index: int):
        """Handle preset selection - show preview of preset values.

        When a preset is selected, update the UI sliders to show what values
        will be applied when 'Apply' is clicked.
        """
        if self._suppress_signals or index < 0:
            return

        preset_combo = self.widgets.get("preset_combo")
        if preset_combo is None:
            return

        preset_name = preset_combo.currentData()
        if not preset_name:
            return

        if not hasattr(self.parent, "material_pbr_service"):
            return

        preset_props = self.parent.material_pbr_service.get_preset_properties(preset_name)
        if preset_props:
            self._show_preset_preview(preset_props)

    def _show_preset_preview(self, props: dict):
        """Show preset property values in the UI sliders/spinboxes.

        Args:
            props: Dict of property values to show
        """
        self._suppress_signals = True
        for prop_name in [
            "roughness",
            "metallic",
            "reflectance",
            "alpha",
            "clearcoat",
            "clearcoat_roughness",
            "anisotropy",
            "emissive_intensity",
            "normal_map_strength",
            "transmission",
            "glass_thickness",
        ]:
            if prop_name in props:
                self._set_property_ui(prop_name, props[prop_name])

        if "color" in props:
            color = props["color"]
            if isinstance(color, (list, tuple)) and len(color) >= 3:
                self._current_color = [color[0], color[1], color[2]]
                self._update_color_button()

        self._suppress_signals = False

    def _on_apply_preset(self):
        """Apply the selected preset to the currently selected material."""
        preset_combo = self.widgets.get("preset_combo")
        material_combo = self.widgets.get("material_combo")

        if preset_combo is None or material_combo is None:
            return

        preset_name = preset_combo.currentData()
        material_type = material_combo.currentText()

        if not preset_name or not material_type:
            logger.warning("No preset or material selected")
            return

        if hasattr(self.parent, "material_pbr_service"):
            success = self.parent.material_pbr_service.apply_preset(preset_name, material_type)
            if success:
                # Reload material properties to show applied values
                self._load_material_properties(material_type)
                # Override profile label to show the manually applied preset
                vis_label = self.widgets.get("visual_preset_label")
                if vis_label is not None:
                    vis_label.setText(f"{preset_name} (manual)")
                    vis_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
                clear_btn = self.widgets.get("clear_profile_btn")
                if clear_btn is not None:
                    clear_btn.setVisible(True)
                logger.info(f"Applied preset '{preset_name}' to '{material_type}'")

    def _on_save_preset(self):
        """Save current settings as a custom preset."""
        from PySide6.QtWidgets import QInputDialog

        preset_name, ok = QInputDialog.getText(self.parent, "Save Preset", "Enter preset name:")

        if ok and preset_name:
            if hasattr(self.parent, "material_pbr_service"):
                success = self.parent.material_pbr_service.save_preset(preset_name)
                if success:
                    logger.info(f"Saved custom preset '{preset_name}'")

    def _on_load_preset(self):
        """Load a custom preset."""
        from PySide6.QtWidgets import QInputDialog

        if not hasattr(self.parent, "material_pbr_service"):
            return

        user_presets = self.parent.material_pbr_service.list_user_presets()
        if not user_presets:
            logger.info("No custom presets found")
            return

        preset_name, ok = QInputDialog.getItem(
            self.parent,
            "Load Preset",
            "Select preset:",
            user_presets,
            0,
            False,
        )

        if ok and preset_name:
            success = self.parent.material_pbr_service.load_preset(preset_name)
            if success:
                # Reload current material properties
                material_combo = self.widgets.get("material_combo")
                # Note: Must use 'is not None' because empty QComboBox is falsy
                if material_combo is not None:
                    material_type = material_combo.currentText()
                    if material_type:
                        self._load_material_properties(material_type)
                logger.info(f"Loaded custom preset '{preset_name}'")

    def refresh_material_list(self):
        """Refresh the list of materials from the scene."""
        material_combo = self.widgets.get("material_combo")
        if material_combo is None:
            return

        current_selection = material_combo.currentText()

        # Block signals while repopulating to avoid triggering multiple loads
        material_combo.blockSignals(True)

        material_combo.clear()

        if hasattr(self.parent, "material_pbr_service"):
            material_types = self.parent.material_pbr_service.get_material_types_in_scene()
            for mat_type in material_types:
                material_combo.addItem(mat_type)

        # Restore selection if still valid, otherwise select first item
        if current_selection:
            index = material_combo.findText(current_selection)
            if index >= 0:
                material_combo.setCurrentIndex(index)

        material_combo.blockSignals(False)

        if material_combo.count() > 0:
            selected_material = material_combo.currentText()
            if selected_material:
                self._load_material_properties(selected_material)
                self._update_visibility_buttons(selected_material)
                self._update_profile_info(selected_material)

    def update_ui_from_viz(self):
        """Update UI state from visualizer."""
        renderer = getattr(self.parent, "renderer", None)
        is_pbr = renderer_capabilities(renderer).pbr
        # Show/hide PBR controls based on renderer capabilities
        self.set_enabled_for_renderer(is_pbr)

        # Refresh material list when scene changes
        self.refresh_material_list()

    def set_enabled_for_renderer(self, is_pbr: bool):
        """Show/hide PBR controls based on renderer type.

        Color picker, visibility buttons, and material selector are always available.
        PBR properties are available when the renderer capability map enables them.
        """
        renderer = getattr(self.parent, "renderer", None)
        capabilities = renderer_capabilities(renderer)

        # Always-available controls (both renderers)
        always_enabled = [
            "material_combo",
            "color_btn",
            "hide_btn",
            "highlight_btn",
        ]

        # PBR-only controls
        pbr_only = [
            "roughness_slider",
            "roughness_spin",
            "metallic_slider",
            "metallic_spin",
            "reflectance_slider",
            "reflectance_spin",
            "alpha_slider",
            "alpha_spin",
            "reset_btn",
            "reset_all_btn",
            "preset_combo",
            "apply_preset_btn",
            "save_preset_btn",
            "load_preset_btn",
            "clearcoat_slider",
            "clearcoat_spin",
            "clearcoat_roughness_slider",
            "clearcoat_roughness_spin",
            "anisotropy_slider",
            "anisotropy_spin",
            "emissive_intensity_slider",
            "emissive_intensity_spin",
            "normal_map_strength_slider",
            "normal_map_strength_spin",
        ]
        feature_controls = {
            "material_clearcoat": [
                "clearcoat_slider",
                "clearcoat_spin",
                "clearcoat_roughness_slider",
                "clearcoat_roughness_spin",
            ],
            "material_anisotropy": ["anisotropy_slider", "anisotropy_spin"],
            "material_emissive": [
                "emissive_intensity_slider",
                "emissive_intensity_spin",
            ],
            "material_normal_map": [
                "normal_map_strength_slider",
                "normal_map_strength_spin",
            ],
            "material_transmission": ["transmission_slider", "transmission_spin"],
            "material_volume_thickness": [
                "glass_thickness_slider",
                "glass_thickness_spin",
            ],
        }

        # Enable always-available controls
        for widget_name in always_enabled:
            widget = self.widgets.get(widget_name)
            if widget is not None and hasattr(widget, "setEnabled"):
                widget.setEnabled(True)

        # Enable/disable PBR controls based on renderer
        for widget_name in pbr_only:
            widget = self.widgets.get(widget_name)
            if widget is not None and hasattr(widget, "setEnabled"):
                widget.setEnabled(is_pbr)
        for capability_name, widget_names in feature_controls.items():
            supported = bool(is_pbr and getattr(capabilities, capability_name))
            for widget_name in widget_names:
                widget = self.widgets.get(widget_name)
                if widget is not None and hasattr(widget, "setEnabled"):
                    widget.setEnabled(supported)
                    if not supported and hasattr(widget, "setToolTip"):
                        widget.setToolTip(
                            "The active renderer does not apply this material feature; "
                            "the value is retained if you switch renderers."
                        )

        # Show/hide the PBR group boxes entirely
        pbr_group = self.widgets.get("pbr_group")
        if pbr_group is not None:
            pbr_group.setVisible(is_pbr)
        advanced_group = self.widgets.get("advanced_pbr_group")
        if advanced_group is not None:
            advanced_group.setVisible(is_pbr)

        # Show profile info only for Open3D/pygfx renderers
        profile_group = self.widgets.get("profile_group")
        if profile_group is not None:
            profile_group.setVisible(is_pbr)
