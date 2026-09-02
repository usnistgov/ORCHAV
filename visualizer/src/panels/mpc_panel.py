"""MPC visibility, filtering, coloring, and legend controls for the Paths tab.

The panel owns Qt widget construction and small presentation helpers. User
intent is handed to the UI controller, while canonical MPC filtering, render
budgeting, and renderer payload construction remain in the pipeline and
renderer layers.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListView,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from shared.logging import get_logger
from shared.statistics.themes import (
    AVAILABLE_CATEGORICAL_COLORMAPS,
    AVAILABLE_CONTINUOUS_COLORMAPS,
    theme_manager,
)

from ..app.theme import current_theme
from ..renderers.protocol import renderer_capabilities
from ..scene.defaults import (
    DEFAULT_MPC_LINE_WIDTH_PX,
    DEFAULT_MPC_POINT_SIZE_PX,
    MPC_LINE_WIDTH_BOUNDS_PX,
    MPC_POINT_SIZE_BOUNDS_PX,
)
from ..services.cache_service import CacheInvalidationScope, invalidate_visualizer_cache
from ..services.mpc_interaction_style_service import (
    build_mpc_type_palette,
    mpc_interaction_legend_entries,
    rgb_to_css_hex,
)
from ..state import MPC_ORDER_VALUES, MPC_TYPE_VALUES
from .base import BasePanel

logger = get_logger("orchav.mpc_panel")


def _get_colormap_gradient_css() -> str:
    """Generate a CSS linear gradient from the current continuous colormap."""
    try:
        cmap_name = theme_manager.current.continuous_colormap

        import matplotlib as mpl

        if hasattr(mpl, "colormaps"):
            cmap = mpl.colormaps.get_cmap(cmap_name)
        else:
            import matplotlib.pyplot as plt

            cmap = plt.cm.get_cmap(cmap_name)

        # Five stops are enough for a compact Qt stylesheet preview.
        stops = []
        for pos in [0.0, 0.25, 0.5, 0.75, 1.0]:
            rgba = cmap(pos)
            r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
            stops.append(f"stop:{pos} #{r:02x}{g:02x}{b:02x}")

        return f"qlineargradient(x1:0, y1:0, x2:1, y2:0, {', '.join(stops)})"

    except (ValueError, AttributeError, ImportError) as e:
        logger.debug(f"Failed to generate colormap gradient: {e}")
        # Keep a deterministic legend if Matplotlib or the theme lookup fails.
        return "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00ff88, stop:0.5 #ffd700, stop:1 #ff4444)"


def _get_colormap_colors_at_positions(positions: list[float]) -> list[str]:
    """Get hex colors from the current continuous colormap at specified positions (0-1)."""
    try:
        cmap_name = theme_manager.current.continuous_colormap

        import matplotlib as mpl

        if hasattr(mpl, "colormaps"):
            cmap = mpl.colormaps.get_cmap(cmap_name)
        else:
            import matplotlib.pyplot as plt

            cmap = plt.cm.get_cmap(cmap_name)

        colors = []
        for pos in positions:
            rgba = cmap(pos)
            hex_color = f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"
            colors.append(hex_color)
        return colors

    except (ValueError, AttributeError, ImportError) as e:
        logger.debug(f"Failed to get colormap colors: {e}")
        # Match the gradient fallback used by the scalar colorbar.
        return ["#00ff88", "#ffd700", "#ff4444"]


def _get_order_palette_colors() -> list[str]:
    """Get reflection order colors from current theme/categorical colormap as hex strings."""
    try:
        from visualizer.src.utils.colors import get_categorical_order_palette

        # Prefer categorical palettes so order colors stay stable across themes.
        cat_palette = get_categorical_order_palette(n_colors=7)
        if cat_palette is not None:
            return [
                f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}" for c in cat_palette
            ]
        # Fall back to semantic theme colors when no named palette is active.
        theme = theme_manager.current
        return [
            f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
            for c in theme.reflection_order
        ]
    except (ValueError, AttributeError, ImportError) as e:
        logger.debug(f"Failed to get order palette colors: {e}")
        # Last-resort colors keep the legend usable in stripped environments.
        return ["#00ff88", "#ffd700", "#ff8c00", "#ff4444", "#9932cc", "#4169e1", "#2f2f2f"]


def _get_type_legend_entries(
    present_types: tuple[int, ...] | None = None,
):
    """Resolve canonical MPC type legend entries from the active palette."""
    try:
        from visualizer.src.utils.colors import get_categorical_type_palette

        palette = get_categorical_type_palette(n_colors=9)
        if palette is None:
            palette = build_mpc_type_palette(theme_manager.current.interaction_type)
        return mpc_interaction_legend_entries(palette, present_types=present_types)
    except (ValueError, AttributeError, ImportError) as e:
        logger.debug(f"Failed to get type palette colors: {e}")
        # Last-resort colors keep the legend usable in stripped environments.
        fallback = build_mpc_type_palette(
            {
                0: (0.0, 0.83, 0.67),
                1: (0.29, 0.56, 0.89),
                2: (0.0, 0.8, 0.0),
                4: (0.61, 0.35, 0.71),
                8: (0.95, 0.55, 0.16),
            }
        )
        return mpc_interaction_legend_entries(fallback, present_types=present_types)


def _get_type_palette_colors() -> list[str]:
    """Get canonical MPC type legend colors as CSS hex strings."""
    return [rgb_to_css_hex(entry.color) for entry in _get_type_legend_entries()]


class MPCVisualizationPanel(BasePanel):
    """Build the Paths-tab control surface for multipath components."""

    def __init__(self, parent_widget):
        """Initialize legend state and subscribe to theme color changes."""
        super().__init__(parent_widget)
        self._current_color_mode = "reflection_order"
        self._present_mpc_type_codes: tuple[int, ...] = ()
        self._updating_material_model = False
        theme_manager.add_listener(self._on_theme_change)

    def _on_theme_change(self, theme) -> None:
        """Handle theme/colormap changes by refreshing the color legend."""
        if hasattr(self, "widgets") and "color_legend_layout" in self.widgets:
            self.update_color_legend(self._current_color_mode)
            logger.debug(f"Refreshed color legend for theme change: {theme.name}")

    def create_panel(self) -> QGroupBox:
        """Create the primary MPC control panel used by the panel manager."""
        group = self.create_group_box("MPCs")

        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._create_visibility_group())
        layout.addWidget(self._create_coloring_group())
        layout.addWidget(self._create_mpc_selection_group())
        layout.addWidget(self._create_render_budget_group())
        layout.addWidget(self._create_markers_group())
        layout.addStretch()
        return group

    def create_range_filters_panel(self) -> QGroupBox:
        """Create range filters as a separate panel for the panel manager."""
        group = self.create_group_box("Advanced Filtering")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._create_range_filters())
        layout.addStretch()
        return group

    def _create_visibility_group(self) -> QGroupBox:
        """Create toggles for MPC lines, bounce markers, and summary text."""
        group = self.create_subgroup_box("Visibility")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._create_mpc_core_controls())
        layout.addWidget(self._create_mpc_info_section())
        return group

    def _create_mpc_selection_group(self) -> QGroupBox:
        """Create preset, discrete, and material-selection filters."""
        group = self.create_subgroup_box("MPC Selection")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._create_preset_controls())
        layout.addWidget(self._create_discrete_filters())
        layout.addWidget(self._create_materials_section())
        return group

    def _create_coloring_group(self) -> QGroupBox:
        """Create color-mode controls and their compact legend."""
        group = self.create_subgroup_box("Coloring")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._create_color_mode_section())
        layout.addWidget(self._create_color_legend_group())
        return group

    def _create_categorical_palette_combo(self) -> QComboBox:
        """Create the categorical palette combo for MPC order/type colors."""
        combo = QComboBox()
        for cmap in AVAILABLE_CATEGORICAL_COLORMAPS:
            display_name = "Semantic (Default)" if cmap == "theme_default" else cmap
            combo.addItem(display_name, cmap)

        current_cat_cmap = theme_manager.current.categorical_colormap
        for i in range(combo.count()):
            if combo.itemData(i) == current_cat_cmap:
                combo.setCurrentIndex(i)
                break

        combo.setToolTip(
            "Colors for MPC reflection order and interaction type rendering.\n"
            "'Semantic' uses meaningful colors such as green for LoS."
        )
        combo.currentIndexChanged.connect(self._on_categorical_cmap_changed)
        self.widgets["categorical_cmap_combo"] = combo
        return combo

    def _create_continuous_palette_combo(self) -> QComboBox:
        """Create the scalar palette combo for MPC delay/loss colors."""
        combo = QComboBox()
        for cmap in AVAILABLE_CONTINUOUS_COLORMAPS:
            combo.addItem(cmap)

        current_cmap = theme_manager.current.continuous_colormap
        index = combo.findText(current_cmap)
        if index >= 0:
            combo.setCurrentIndex(index)

        combo.setToolTip("Colors for scalar MPC modes such as delay and path loss.")
        combo.currentTextChanged.connect(self._on_continuous_cmap_changed)
        self.widgets["continuous_cmap_combo"] = combo
        return combo

    def _create_render_budget_group(self) -> QGroupBox:
        """Create controls that cap how many strongest MPCs are rendered."""
        group = self.create_subgroup_box("Render Budget")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._create_topk_render_controls())
        return group

    def _on_categorical_cmap_changed(self, index: int) -> None:
        """Change the categorical palette used for MPC order/type colors."""
        combo = self.widgets.get("categorical_cmap_combo")
        if combo is None:
            return
        cmap_name = combo.itemData(index)
        if cmap_name is None:
            return
        try:
            theme_manager.set_colormap("categorical", cmap_name)
            logger.info("MPC categorical colormap changed to: %s", cmap_name)
            if hasattr(self.parent, "request_redraw"):
                self.parent.request_redraw()
        except ValueError as exc:
            logger.warning("Failed to set categorical colormap: %s", exc)

    def _on_continuous_cmap_changed(self, cmap_name: str) -> None:
        """Change the scalar palette used for MPC delay/path-loss colors."""
        try:
            theme_manager.set_colormap("continuous", cmap_name)
            logger.info("MPC scalar colormap changed to: %s", cmap_name)
            if hasattr(self.parent, "request_redraw"):
                self.parent.request_redraw()
        except ValueError as exc:
            logger.warning("Failed to set scalar colormap: %s", exc)

    def _create_markers_group(self) -> QGroupBox:
        """Create Paths-owned MPC marker and path-thickness controls."""
        group = self.create_subgroup_box("Marker")
        layout = QGridLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        self.widgets["mpc_interaction_markers_cb"] = QCheckBox("Interaction Markers")
        self.widgets["mpc_interaction_markers_cb"].setToolTip(
            "Render physical MPC interaction points as glyphs:\n"
            "circle = specular, triangle = diffuse, diamond = refraction,\n"
            "plus = diffraction, square = virtual, cross = unknown.\n"
            "LoS has no interaction point."
        )
        self.widgets["mpc_interaction_markers_cb"].setStyleSheet("QCheckBox { font-size: 11px; }")
        layout.addWidget(self.widgets["mpc_interaction_markers_cb"], 0, 0, 1, 2)

        point_label = QLabel("Point / marker size:")
        point_label.setToolTip("Size of bounce points and interaction-marker glyphs in pixels.")
        layout.addWidget(point_label, 1, 0)
        self.widgets["point_size_spin"] = QDoubleSpinBox()
        self.widgets["point_size_spin"].setDecimals(2)
        self.widgets["point_size_spin"].setRange(*MPC_POINT_SIZE_BOUNDS_PX)
        self.widgets["point_size_spin"].setSingleStep(0.5)
        self.widgets["point_size_spin"].setValue(DEFAULT_MPC_POINT_SIZE_PX)
        self.widgets["point_size_spin"].setSuffix(" px")
        self.widgets["point_size_spin"].setKeyboardTracking(False)
        self.widgets["point_size_spin"].setToolTip(point_label.toolTip())
        self.widgets["point_size_spin"].valueChanged.connect(self._on_point_size_changed)
        layout.addWidget(self.widgets["point_size_spin"], 1, 1)

        line_label = QLabel("Path line width:")
        line_label.setToolTip("Width of MPC ray-path lines in pixels.")
        layout.addWidget(line_label, 2, 0)
        self.widgets["line_width_spin"] = QDoubleSpinBox()
        self.widgets["line_width_spin"].setDecimals(2)
        self.widgets["line_width_spin"].setRange(*MPC_LINE_WIDTH_BOUNDS_PX)
        self.widgets["line_width_spin"].setSingleStep(0.5)
        self.widgets["line_width_spin"].setValue(DEFAULT_MPC_LINE_WIDTH_PX)
        self.widgets["line_width_spin"].setSuffix(" px")
        self.widgets["line_width_spin"].setKeyboardTracking(False)
        self.widgets["line_width_spin"].setToolTip(line_label.toolTip())
        self.widgets["line_width_spin"].valueChanged.connect(self._on_line_width_changed)
        layout.addWidget(self.widgets["line_width_spin"], 2, 1)

        layout.setColumnStretch(0, 1)
        self.refresh_renderer_controls_state()
        return group

    def _create_mpc_core_controls(self) -> QWidget:
        """Create Paths-local visibility toggles for MPC presentation details."""
        group = QWidget()
        layout = QHBoxLayout(group)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)

        self.widgets["mpc_paths_cb"] = QCheckBox("Paths")
        self.widgets["mpc_paths_cb"].setChecked(True)
        self.widgets["mpc_paths_cb"].setToolTip("Show MPC path segments.")
        self.widgets["mpc_paths_cb"].setStyleSheet("QCheckBox { font-size: 11px; }")
        layout.addWidget(self.widgets["mpc_paths_cb"])

        self.widgets["mpc_bounce_points_cb"] = QCheckBox("Bounce Points")
        self.widgets["mpc_bounce_points_cb"].setChecked(True)
        self.widgets["mpc_bounce_points_cb"].setToolTip(
            "Show physical interaction points without TX/RX endpoints."
        )
        self.widgets["mpc_bounce_points_cb"].setStyleSheet("QCheckBox { font-size: 11px; }")
        layout.addWidget(self.widgets["mpc_bounce_points_cb"])

        layout.addStretch()
        return group

    def _renderer_supports_interaction_markers(self) -> bool:
        """Return whether the active renderer can draw interaction markers."""
        parent = getattr(self, "parent", None)
        renderer = getattr(parent, "renderer", None)
        return renderer_capabilities(renderer).mpc_type_markers

    def refresh_renderer_controls_state(self) -> None:
        """Refresh renderer-specific MPC display controls."""
        markers_cb = self.widgets.get("mpc_interaction_markers_cb")
        if markers_cb is not None:
            markers_cb.setVisible(self._renderer_supports_interaction_markers())

        colormap_group = self.widgets.get("colormap_group")
        if colormap_group is not None:
            colormap_group.setVisible(True)

        state = getattr(getattr(self, "parent", None), "app_state", None)
        if state is not None and markers_cb is not None:
            markers_cb.blockSignals(True)
            markers_cb.setChecked(bool(getattr(state, "show_mpc_type_markers", False)))
            markers_cb.blockSignals(False)

    @staticmethod
    def _finite_size_value(
        value: object,
        bounds: tuple[float, float],
    ) -> float | None:
        """Return a finite numeric size within the shared renderer safety bounds."""
        if isinstance(value, bool):
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric) or not bounds[0] <= numeric <= bounds[1]:
            return None
        return numeric

    def _apply_renderer_size(self, setter_name: str, value: float) -> None:
        """Apply one Paths-owned MPC appearance value to the active renderer."""
        renderer = getattr(getattr(self, "parent", None), "renderer", None)
        setter = getattr(renderer, setter_name, None)
        if callable(setter):
            setter(float(value))

    def _on_point_size_changed(self, value: float) -> None:
        """Apply bounce-point and interaction-marker size."""
        self._apply_renderer_size("set_point_size", value)

    def _on_line_width_changed(self, value: float) -> None:
        """Apply MPC ray-path line width."""
        self._apply_renderer_size("set_line_width", value)

    def restore_session_state(self, state: dict[str, object]) -> None:
        """Restore pre-split MPC size keys through the Paths-owned controls."""
        for state_key, widget_key, setter_name in (
            ("point_size", "point_size_spin", "set_point_size"),
            ("mpc_line_width", "line_width_spin", "set_line_width"),
        ):
            widget = self.widgets.get(widget_key)
            if widget is None or state_key not in state:
                continue
            value = self._finite_size_value(
                state[state_key],
                (widget.minimum(), widget.maximum()),
            )
            if value is None:
                continue
            with QSignalBlocker(widget):
                widget.setValue(value)
            self._apply_renderer_size(setter_name, value)

    def _create_topk_render_controls(self) -> QWidget:
        """Create optional strongest-MPC render cap controls."""
        group = QWidget()
        main_layout = QVBoxLayout(group)
        main_layout.setSpacing(4)
        main_layout.setContentsMargins(0, 0, 0, 0)

        layout = QHBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        self.widgets["topk_render_cb"] = QCheckBox("Limit rendered MPCs")
        self.widgets["topk_render_cb"].setChecked(False)
        self.widgets["topk_render_cb"].setToolTip(
            "Limit rendered MPCs to the K strongest components after filters."
        )
        self.widgets["topk_render_cb"].setStyleSheet("QCheckBox { font-size: 10px; }")
        layout.addWidget(self.widgets["topk_render_cb"])

        label = QLabel("Max MPCs:")
        label.setStyleSheet("font-size: 10px;")
        layout.addWidget(label)

        self.widgets["topk_render_max_spin"] = QSpinBox()
        self.widgets["topk_render_max_spin"].setRange(1, 2_000_000)
        self.widgets["topk_render_max_spin"].setSingleStep(1000)
        self.widgets["topk_render_max_spin"].setValue(20000)
        self.widgets["topk_render_max_spin"].setEnabled(False)
        self.widgets["topk_render_max_spin"].setStyleSheet("QSpinBox { font-size: 10px; }")
        self.widgets["topk_render_max_spin"].setToolTip(
            "Maximum number of strongest MPCs to render when the cap is enabled."
        )
        layout.addWidget(self.widgets["topk_render_max_spin"])

        self.widgets["topk_render_cb"].toggled.connect(
            self.widgets["topk_render_max_spin"].setEnabled
        )
        layout.addStretch()
        main_layout.addLayout(layout)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(4)
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_label = QLabel("Quick:")
        preset_label.setStyleSheet("font-size: 9px;")
        preset_row.addWidget(preset_label)
        self.widgets["topk_render_preset_buttons"] = []
        for value, label_text in (
            (10, "10"),
            (100, "100"),
            (1000, "1k"),
            (5000, "5k"),
            (20000, "20k"),
            (100000, "100k"),
        ):
            button = QPushButton(label_text)
            button.setStyleSheet(
                "QPushButton { font-size: 9px; padding: 2px 6px; min-width: 28px; }"
            )
            button.setToolTip(f"Set rendered MPC cap to {value:,}")
            button.clicked.connect(
                lambda _checked=False, v=value: self.widgets["topk_render_max_spin"].setValue(v)
            )
            preset_row.addWidget(button)
            self.widgets["topk_render_preset_buttons"].append(button)
        preset_row.addStretch()
        main_layout.addLayout(preset_row)
        return group

    def _create_color_mode_section(self) -> QWidget:
        """Create color mode and palette controls in compact rows."""
        container = QWidget()
        self.widgets["color_mode_section"] = container

        layout = QVBoxLayout(container)
        layout.setSpacing(3)
        layout.setContentsMargins(0, 0, 0, 0)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(2)
        mode_row.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Color By:")
        header.setStyleSheet("font-size: 10px;")
        self.widgets["color_mode_label"] = header
        mode_row.addWidget(header)

        self.widgets["color_mode_group"] = QButtonGroup()

        radio_style = "QRadioButton { font-size: 9px; margin: 0px; padding: 0px; }"

        radio_defs = [
            ("reflection_order_rb", "Order", None, True),
            ("mpc_type_rb", "Type", None, False),
            ("delay_rb", "Delay", None, False),
            ("path_loss_rb", "Loss", None, False),
            ("material_rb", "Material", "Color MPC segments by bounce material", False),
            (
                "reconstruction_type_rb",
                "Recon.",
                "Show geometry-resolved vs virtual bounces",
                False,
            ),
        ]

        for key, label, tooltip, checked in radio_defs:
            rb = QRadioButton(label)
            rb.setStyleSheet(radio_style)
            rb.setChecked(checked)
            if tooltip:
                rb.setToolTip(tooltip)
            if key == "reconstruction_type_rb":
                rb.setVisible(False)
            self.widgets[key] = rb
            self.widgets["color_mode_group"].addButton(rb)
            mode_row.addWidget(rb)

        mode_row.addStretch()
        layout.addLayout(mode_row)

        colormap_group = self.create_subgroup_box("Colormap")
        colormap_group.setToolTip(
            "Select palettes for categorical MPC modes and scalar delay/path-loss modes."
        )
        self.widgets["colormap_group"] = colormap_group
        palette_row = QHBoxLayout(colormap_group)
        palette_row.setSpacing(6)
        palette_row.setContentsMargins(6, 4, 6, 4)

        palette_label = QLabel("MPC:")
        palette_label.setStyleSheet("font-size: 10px;")
        self.widgets["categorical_cmap_label"] = palette_label
        palette_row.addWidget(palette_label)

        categorical_combo = self._create_categorical_palette_combo()
        categorical_combo.setMaximumWidth(135)
        palette_row.addWidget(categorical_combo)

        scalar_label = QLabel("Scalar:")
        scalar_label.setStyleSheet("font-size: 10px;")
        self.widgets["continuous_cmap_label"] = scalar_label
        palette_row.addWidget(scalar_label)

        continuous_combo = self._create_continuous_palette_combo()
        continuous_combo.setMaximumWidth(105)
        palette_row.addWidget(continuous_combo)

        palette_row.addStretch()
        layout.addWidget(colormap_group)
        return container

    def _create_mpc_info_section(self) -> QWidget:
        """Create compact MPC count and lazy Explorer entry point."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.widgets["mpc_info_label"] = QLabel("MPCs: --")
        self.widgets["mpc_info_label"].setObjectName("mpcInfoLabel")
        self.widgets["mpc_info_label"].setStyleSheet("""
            font-size: 10px;
            font-weight: normal;
        """)
        self.widgets["mpc_info_label"].setWordWrap(True)
        layout.addWidget(self.widgets["mpc_info_label"], 1)

        explorer_button = QPushButton("Open MPC Explorer...")
        explorer_button.setObjectName("openMpcExplorerButton")
        explorer_button.setToolTip(
            "Open the scalable path table and selected-path inspection window."
        )
        self.widgets["mpc_explorer_btn"] = explorer_button
        layout.addWidget(explorer_button)
        return container

    def _create_discrete_filters(self) -> QWidget:
        """Create orders and types filter rows with inline labels."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(3)
        layout.setContentsMargins(0, 0, 0, 0)

        label_style = "font-weight: bold; font-size: 10px;"
        cb_style = "QCheckBox { font-size: 9px; margin: 0px; padding: 0px; }"

        # Orders row
        orders_row = QHBoxLayout()
        orders_row.setSpacing(4)
        orders_row.setContentsMargins(0, 0, 0, 0)
        orders_label = QLabel("Orders:")
        orders_label.setStyleSheet(label_style)
        orders_label.setFixedWidth(50)
        orders_row.addWidget(orders_label)

        order_labels = ("LoS", "1st", "2nd", "3rd", "4th", "5th", "6+")
        for i, label in zip(MPC_ORDER_VALUES, order_labels):
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            checkbox.setStyleSheet(cb_style)
            self.widgets[f"order_{i}_cb"] = checkbox
            orders_row.addWidget(checkbox)
        orders_row.addStretch()
        layout.addLayout(orders_row)

        # Types row
        types_row = QHBoxLayout()
        types_row.setSpacing(4)
        types_row.setContentsMargins(0, 0, 0, 0)
        types_label = QLabel("Types:")
        types_label.setStyleSheet(label_style)
        types_label.setFixedWidth(50)
        types_row.addWidget(types_label)

        type_labels = ("LoS", "Spec", "Diff", "Refr", "Diffr", "Virtual")
        for label, value in zip(type_labels, MPC_TYPE_VALUES):
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            checkbox.setStyleSheet(cb_style)
            self.widgets[f"type_{value}_cb"] = checkbox
            types_row.addWidget(checkbox)
        types_row.addStretch()
        layout.addLayout(types_row)

        return container

    def _create_materials_section(self) -> QWidget:
        """Create materials filter section with label and buttons on same row."""
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setSpacing(3)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Header row: label + bulk action buttons
        header_row = QHBoxLayout()
        header_row.setSpacing(6)
        header_row.setContentsMargins(0, 0, 0, 0)
        materials_label = QLabel("Materials:")
        materials_label.setStyleSheet("font-weight: bold; font-size: 10px;")
        header_row.addWidget(materials_label)

        distinct_cb = QCheckBox("Distinct")
        distinct_cb.setToolTip(
            "Use auto-generated maximally distinct colors instead of ITU palette"
        )
        distinct_cb.setChecked(False)
        distinct_cb.setStyleSheet("font-size: 9px;")
        distinct_cb.setVisible(False)  # Hidden until material color mode is selected
        self.widgets["distinct_material_colors_cb"] = distinct_cb
        header_row.addWidget(distinct_cb)

        header_row.addLayout(self._create_material_bulk_actions())
        main_layout.addLayout(header_row)

        model = QStandardItemModel(container)
        model.itemChanged.connect(self._on_material_item_changed)
        self.widgets["materials_model"] = model

        material_view = QListView()
        material_view.setObjectName("materialFilterList")
        material_view.setModel(model)
        material_view.setUniformItemSizes(True)
        material_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        material_view.setSelectionMode(QAbstractItemView.NoSelection)
        material_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        material_view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        material_view.setToolTip("Select which bounce materials are included in MPC rendering.")
        self.widgets["materials_view"] = material_view
        main_layout.addWidget(material_view)

        self.set_materials([], checked=set())

        return container

    def _create_range_filters(self) -> QWidget:
        """Create filter controls for delay, path loss, angles, and previews."""
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setSpacing(4)
        main_layout.setContentsMargins(0, 0, 0, 0)

        spin_style = "QDoubleSpinBox { font-size: 9px; max-width: 70px; }"
        label_style = "font-size: 9px;"

        def make_spin(
            suffix: str, min_val: float, max_val: float, decimals: int, tooltip: str
        ) -> QDoubleSpinBox:
            """Create a range-filter spin box with the domain unit suffix."""
            spin = QDoubleSpinBox()
            spin.setDecimals(decimals)
            spin.setRange(min_val, max_val)
            spin.setValue(min_val)
            spin.setSuffix(f" {suffix}" if suffix else "")
            spin.setStyleSheet(spin_style)
            spin.setToolTip(tooltip)
            return spin

        def add_range_row(
            layout: QGridLayout,
            row: int,
            label: str,
            key_prefix: str,
            suffix: str,
            min_val: float,
            max_val: float,
            tooltip_name: str,
        ) -> None:
            """Add a min/max row whose endpoints mean no active filter."""
            layout.addWidget(QLabel(label), row, 0)
            min_spin = make_spin(
                suffix,
                min_val,
                max_val,
                1,
                f"Minimum {tooltip_name} (leave at min for no filter)",
            )
            min_spin.setSpecialValueText("Min")
            self.widgets[f"{key_prefix}_filter_min"] = min_spin
            layout.addWidget(min_spin, row, 1)
            separator = QLabel("to")
            separator.setStyleSheet(label_style)
            layout.addWidget(separator, row, 2)
            max_spin = make_spin(
                suffix,
                min_val,
                max_val,
                1,
                f"Maximum {tooltip_name} (leave at max for no filter)",
            )
            max_spin.setValue(max_val)
            max_spin.setSpecialValueText("Max")
            self.widgets[f"{key_prefix}_filter_max"] = max_spin
            layout.addWidget(max_spin, row, 3)

        delay_group = self.create_subgroup_box("Delay")
        delay_layout = QGridLayout(delay_group)
        delay_layout.setSpacing(4)
        delay_layout.setContentsMargins(6, 6, 6, 6)
        add_range_row(delay_layout, 0, "Delay:", "delay", "ns", 0.0, 10000.0, "delay")
        delay_layout.setColumnStretch(4, 1)

        path_loss_group = self.create_subgroup_box("Path Loss")
        path_loss_layout = QGridLayout(path_loss_group)
        path_loss_layout.setSpacing(4)
        path_loss_layout.setContentsMargins(6, 6, 6, 6)
        add_range_row(
            path_loss_layout,
            0,
            "Loss:",
            "power",
            "dB",
            0.0,
            200.0,
            "path loss",
        )
        path_loss_layout.setColumnStretch(4, 1)

        scalar_row = QWidget()
        scalar_layout = QHBoxLayout(scalar_row)
        scalar_layout.setSpacing(4)
        scalar_layout.setContentsMargins(0, 0, 0, 0)
        scalar_layout.addWidget(delay_group, 1)
        scalar_layout.addWidget(path_loss_group, 1)
        main_layout.addWidget(scalar_row)

        angle_group = self.create_subgroup_box("Angles")
        angle_layout = QVBoxLayout(angle_group)
        angle_layout.setSpacing(4)
        angle_layout.setContentsMargins(6, 6, 6, 6)
        angle_grid = QGridLayout()
        angle_grid.setSpacing(4)
        angle_grid.setContentsMargins(0, 0, 0, 0)

        angle_defs = [
            (0, 0, "AoD Az:", "\u00b0", -180.0, 180.0, "aod_az"),
            (0, 5, "AoD El:", "\u00b0", -90.0, 90.0, "aod_el"),
            (1, 0, "AoA Az:", "\u00b0", -180.0, 180.0, "aoa_az"),
            (1, 5, "AoA El:", "\u00b0", -90.0, 90.0, "aoa_el"),
        ]

        for row, col_base, lbl, suffix, mn, mx, key in angle_defs:
            angle_grid.addWidget(QLabel(lbl), row, col_base)
            s_min = make_spin(suffix, mn, mx, 1, f"Minimum {lbl.strip(':')}")
            s_min.setSpecialValueText("Min")
            self.widgets[f"{key}_filter_min"] = s_min
            angle_grid.addWidget(s_min, row, col_base + 1)
            dash = QLabel("to")
            dash.setStyleSheet(label_style)
            angle_grid.addWidget(dash, row, col_base + 2)
            s_max = make_spin(suffix, mn, mx, 1, f"Maximum {lbl.strip(':')}")
            s_max.setValue(mx)
            s_max.setSpecialValueText("Max")
            self.widgets[f"{key}_filter_max"] = s_max
            angle_grid.addWidget(s_max, row, col_base + 3)

        angle_grid.setColumnMinimumWidth(4, 12)
        angle_grid.setColumnStretch(9, 1)
        angle_layout.addLayout(angle_grid)
        self.widgets["angle_filters_container"] = angle_group
        main_layout.addWidget(angle_group)

        preview_group = self.create_subgroup_box("Angular Preview")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setSpacing(4)
        preview_layout.setContentsMargins(6, 6, 6, 6)
        self.widgets["angular_preview_group"] = preview_group

        preview_controls = QWidget()
        footer = QHBoxLayout(preview_controls)
        footer.setSpacing(6)
        footer.setContentsMargins(0, 0, 0, 0)

        aod_cb = QCheckBox("AoD")
        aod_cb.setChecked(False)
        aod_cb.setStyleSheet(
            "QCheckBox { font-size: 9px; }"
            "QCheckBox::indicator:checked { background-color: #ff4d33; border: 1px solid #cc3d29; }"
        )
        aod_cb.setToolTip("Show the filled AoD filter sector at the TX selected in Context")
        self.widgets["show_aod_aperture_cb"] = aod_cb
        footer.addWidget(aod_cb)

        aoa_cb = QCheckBox("AoA")
        aoa_cb.setChecked(False)
        aoa_cb.setStyleSheet(
            "QCheckBox { font-size: 9px; }"
            "QCheckBox::indicator:checked { background-color: #3380ff; border: 1px solid #2266cc; }"
        )
        aoa_cb.setToolTip("Show the filled AoA filter sector at the RX selected in Context")
        self.widgets["show_aoa_aperture_cb"] = aoa_cb
        footer.addWidget(aoa_cb)

        global_ref_cb = QCheckBox("Global")
        global_ref_cb.setChecked(False)
        global_ref_cb.setStyleSheet("QCheckBox { font-size: 9px; }")
        global_ref_cb.setToolTip("Show global angular axes at the TX/RX nodes selected in Context")
        self.widgets["show_global_angular_reference_cb"] = global_ref_cb
        footer.addWidget(global_ref_cb)

        local_ref_cb = QCheckBox("Local")
        local_ref_cb.setChecked(False)
        local_ref_cb.setStyleSheet("QCheckBox { font-size: 9px; }")
        local_ref_cb.setToolTip(
            "Show local device angular axes at the TX/RX nodes selected in Context"
        )
        self.widgets["show_local_angular_reference_cb"] = local_ref_cb
        footer.addWidget(local_ref_cb)

        radius_label = QLabel("Radius:")
        radius_label.setStyleSheet("font-size: 9px;")
        self.widgets["aperture_radius_label"] = radius_label
        footer.addWidget(radius_label)

        radius_spin = QDoubleSpinBox()
        radius_spin.setDecimals(1)
        radius_spin.setRange(0.5, 50.0)
        radius_spin.setValue(5.0)
        radius_spin.setSuffix(" m")
        radius_spin.setStyleSheet("QDoubleSpinBox { font-size: 9px; max-width: 70px; }")
        radius_spin.setToolTip("Radius of aperture visualization in meters")
        self.widgets["aperture_radius_spin"] = radius_spin
        footer.addWidget(radius_spin)

        footer.addStretch()
        self.widgets["aperture_controls_container"] = preview_controls
        preview_layout.addWidget(preview_controls)

        selection_label = QLabel("")
        selection_label.setWordWrap(True)
        selection_label.setStyleSheet("font-size: 9px;")
        self.widgets["aperture_selection_label"] = selection_label
        preview_layout.addWidget(selection_label)

        status_row = QWidget()
        status_layout = QHBoxLayout(status_row)
        status_layout.setSpacing(8)
        status_layout.setContentsMargins(0, 0, 0, 0)

        aod_status_label = QLabel("")
        aod_status_label.setWordWrap(True)
        aod_status_label.setStyleSheet("font-size: 9px;")
        self.widgets["aod_aperture_status_label"] = aod_status_label
        status_layout.addWidget(aod_status_label, 1)

        aoa_status_label = QLabel("")
        aoa_status_label.setWordWrap(True)
        aoa_status_label.setStyleSheet("font-size: 9px;")
        self.widgets["aoa_aperture_status_label"] = aoa_status_label
        status_layout.addWidget(aoa_status_label, 1)
        self.widgets["aperture_status_row"] = status_row
        preview_layout.addWidget(status_row)

        # Kept for older tests/session code paths that look up a combined status.
        status_label = QLabel("")
        status_label.setVisible(False)
        self.widgets["aperture_status_label"] = status_label
        main_layout.addWidget(preview_group)

        actions = QWidget()
        actions_layout = QHBoxLayout(actions)
        actions_layout.setSpacing(6)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.addStretch()
        reset_btn = QPushButton("Reset Filters")
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                padding: 3px 8px;
                border-radius: 3px;
                font-size: 9px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #7f8c8d; }
            """)
        reset_btn.setToolTip("Reset all advanced filters to show all MPCs")
        reset_btn.clicked.connect(self._on_reset_range_filters)
        self.widgets["reset_range_filters_btn"] = reset_btn
        actions_layout.addWidget(reset_btn)
        main_layout.addWidget(actions)

        # Connect aperture control signals
        aod_cb.toggled.connect(self._on_aod_aperture_toggled)
        aoa_cb.toggled.connect(self._on_aoa_aperture_toggled)
        global_ref_cb.toggled.connect(self._on_global_angular_reference_toggled)
        local_ref_cb.toggled.connect(self._on_local_angular_reference_toggled)
        radius_spin.valueChanged.connect(self._on_aperture_radius_changed)

        # Connect all spin box signals
        for widget_name in [
            "delay_filter_min",
            "delay_filter_max",
            "power_filter_min",
            "power_filter_max",
            "aoa_az_filter_min",
            "aoa_az_filter_max",
            "aoa_el_filter_min",
            "aoa_el_filter_max",
            "aod_az_filter_min",
            "aod_az_filter_max",
            "aod_el_filter_min",
            "aod_el_filter_max",
        ]:
            self.widgets[widget_name].valueChanged.connect(self._emit_range_filter_changed)

        self.refresh_aperture_preview_state()

        return container

    def _on_reset_range_filters(self):
        """Reset all range filter spin boxes to their min/max values."""
        self.widgets["delay_filter_min"].setValue(self.widgets["delay_filter_min"].minimum())
        self.widgets["delay_filter_max"].setValue(self.widgets["delay_filter_max"].maximum())

        self.widgets["power_filter_min"].setValue(self.widgets["power_filter_min"].minimum())
        self.widgets["power_filter_max"].setValue(self.widgets["power_filter_max"].maximum())

        self.widgets["aoa_az_filter_min"].setValue(self.widgets["aoa_az_filter_min"].minimum())
        self.widgets["aoa_az_filter_max"].setValue(self.widgets["aoa_az_filter_max"].maximum())
        self.widgets["aoa_el_filter_min"].setValue(self.widgets["aoa_el_filter_min"].minimum())
        self.widgets["aoa_el_filter_max"].setValue(self.widgets["aoa_el_filter_max"].maximum())
        self.widgets["aod_az_filter_min"].setValue(self.widgets["aod_az_filter_min"].minimum())
        self.widgets["aod_az_filter_max"].setValue(self.widgets["aod_az_filter_max"].maximum())
        self.widgets["aod_el_filter_min"].setValue(self.widgets["aod_el_filter_min"].minimum())
        self.widgets["aod_el_filter_max"].setValue(self.widgets["aod_el_filter_max"].maximum())

        # Emit filter changed signal if parent visualizer is available
        if hasattr(self, "parent") and self.parent:
            self._emit_range_filter_changed()
        self.refresh_aperture_preview_state()

    def _emit_range_filter_changed(self):
        """Emit range filter changed signal to update visualization."""
        if not hasattr(self, "parent") or not self.parent:
            self.refresh_aperture_preview_state()
            return

        delay_min = self.widgets["delay_filter_min"].value()
        delay_max = self.widgets["delay_filter_max"].value()
        delay_min = None if delay_min == self.widgets["delay_filter_min"].minimum() else delay_min
        delay_max = None if delay_max == self.widgets["delay_filter_max"].maximum() else delay_max

        power_min = self.widgets["power_filter_min"].value()
        power_max = self.widgets["power_filter_max"].value()
        power_min = None if power_min == self.widgets["power_filter_min"].minimum() else power_min
        power_max = None if power_max == self.widgets["power_filter_max"].maximum() else power_max

        aoa_az_min = self.widgets["aoa_az_filter_min"].value()
        aoa_az_max = self.widgets["aoa_az_filter_max"].value()
        aoa_az_min = (
            None if aoa_az_min == self.widgets["aoa_az_filter_min"].minimum() else aoa_az_min
        )
        aoa_az_max = (
            None if aoa_az_max == self.widgets["aoa_az_filter_max"].maximum() else aoa_az_max
        )

        aoa_el_min = self.widgets["aoa_el_filter_min"].value()
        aoa_el_max = self.widgets["aoa_el_filter_max"].value()
        aoa_el_min = (
            None if aoa_el_min == self.widgets["aoa_el_filter_min"].minimum() else aoa_el_min
        )
        aoa_el_max = (
            None if aoa_el_max == self.widgets["aoa_el_filter_max"].maximum() else aoa_el_max
        )

        aod_az_min = self.widgets["aod_az_filter_min"].value()
        aod_az_max = self.widgets["aod_az_filter_max"].value()
        aod_az_min = (
            None if aod_az_min == self.widgets["aod_az_filter_min"].minimum() else aod_az_min
        )
        aod_az_max = (
            None if aod_az_max == self.widgets["aod_az_filter_max"].maximum() else aod_az_max
        )

        aod_el_min = self.widgets["aod_el_filter_min"].value()
        aod_el_max = self.widgets["aod_el_filter_max"].value()
        aod_el_min = (
            None if aod_el_min == self.widgets["aod_el_filter_min"].minimum() else aod_el_min
        )
        aod_el_max = (
            None if aod_el_max == self.widgets["aod_el_filter_max"].maximum() else aod_el_max
        )

        if hasattr(self.parent, "ui_controller"):
            self.parent.ui_controller.handle_range_filter_changed(
                delay_min_ns=delay_min,
                delay_max_ns=delay_max,
                power_min_db=power_min,
                power_max_db=power_max,
                aoa_az_min_deg=aoa_az_min,
                aoa_az_max_deg=aoa_az_max,
                aoa_el_min_deg=aoa_el_min,
                aoa_el_max_deg=aoa_el_max,
                aod_az_min_deg=aod_az_min,
                aod_az_max_deg=aod_az_max,
                aod_el_min_deg=aod_el_min,
                aod_el_max_deg=aod_el_max,
            )
        self.refresh_aperture_preview_state()

    @staticmethod
    def _is_specific_node_selection(value) -> bool:
        """Return whether a TX/RX selector names one concrete node."""
        if value is None:
            return False
        if isinstance(value, str):
            return value not in {"all", ""}
        return True

    @staticmethod
    def _format_node_selection(value, prefix: str) -> str:
        """Format a TX/RX selector value for aperture-preview status text."""
        if value in {"all", None, ""}:
            return "All"
        if isinstance(value, int):
            return f"{prefix}{value + 1}"
        return str(value)

    def _aperture_selection_state(self) -> tuple[object, object]:
        """Read selected TX/RX values from app state for angular previews."""
        state = getattr(getattr(self, "parent", None), "app_state", None)
        if state is None:
            return "all", "all"
        return getattr(state, "selected_tx", "all"), getattr(state, "selected_rx", "all")

    def _renderer_supports_angular_preview(self) -> bool:
        """Return whether the active renderer supports angular preview glyphs."""
        renderer = getattr(getattr(self, "parent", None), "renderer", None)
        if renderer is None:
            return True
        return renderer_capabilities(renderer).angular_preview

    def refresh_aperture_preview_state(self) -> None:
        """Refresh aperture preview controls and user-visible eligibility status."""
        self.refresh_renderer_controls_state()

        aoa_cb = self.widgets.get("show_aoa_aperture_cb")
        aod_cb = self.widgets.get("show_aod_aperture_cb")
        global_ref_cb = self.widgets.get("show_global_angular_reference_cb")
        local_ref_cb = self.widgets.get("show_local_angular_reference_cb")
        radius_spin = self.widgets.get("aperture_radius_spin")
        radius_label = self.widgets.get("aperture_radius_label")
        preview_group = self.widgets.get("angular_preview_group")
        selection_label = self.widgets.get("aperture_selection_label")
        status_label = self.widgets.get("aperture_status_label")
        status_row = self.widgets.get("aperture_status_row")
        aoa_status_label = self.widgets.get("aoa_aperture_status_label")
        aod_status_label = self.widgets.get("aod_aperture_status_label")
        if aoa_cb is None or aod_cb is None or radius_spin is None or status_label is None:
            return

        if "angle_filters_container" in self.widgets:
            self.widgets["angle_filters_container"].setVisible(True)

        if radius_label is not None:
            radius_label.setEnabled(True)

        selected_tx, selected_rx = self._aperture_selection_state()
        aoa_ready = self._is_specific_node_selection(selected_rx)
        aod_ready = self._is_specific_node_selection(selected_tx)
        reference_ready = aoa_ready or aod_ready
        renderer_ready = self._renderer_supports_angular_preview()

        if preview_group is not None:
            preview_group.setVisible(renderer_ready)
        if not renderer_ready:
            aoa_cb.setEnabled(False)
            aod_cb.setEnabled(False)
            if global_ref_cb is not None:
                global_ref_cb.setEnabled(False)
            if local_ref_cb is not None:
                local_ref_cb.setEnabled(False)
            radius_spin.setEnabled(False)
            status_label.setText("Angular preview is available in pygfx only")
            status_label.setVisible(False)
            return

        aoa_cb.setEnabled(renderer_ready and aoa_ready)
        aod_cb.setEnabled(renderer_ready and aod_ready)
        if global_ref_cb is not None:
            global_ref_cb.setEnabled(renderer_ready and reference_ready)
            global_ref_cb.setToolTip(
                "Angular previews are currently available in pygfx only"
                if not renderer_ready
                else (
                    "Select one TX or RX in Context to enable this reference"
                    if not reference_ready
                    else "Show global angular axes at the TX/RX nodes selected in Context"
                )
            )
        if local_ref_cb is not None:
            local_ref_cb.setEnabled(renderer_ready and reference_ready)
            local_ref_cb.setToolTip(
                "Angular previews are currently available in pygfx only"
                if not renderer_ready
                else (
                    "Select one TX or RX in Context to enable this reference"
                    if not reference_ready
                    else "Show local device angular axes at the TX/RX nodes selected in Context"
                )
            )
        radius_spin.setEnabled(renderer_ready and reference_ready)
        aoa_cb.setToolTip(
            "Angular previews are currently available in pygfx only"
            if not renderer_ready
            else (
                "Select one RX in Context to enable this preview"
                if not aoa_ready
                else "Show the filled AoA filter sector at the RX selected in Context"
            )
        )
        aod_cb.setToolTip(
            "Angular previews are currently available in pygfx only"
            if not renderer_ready
            else (
                "Select one TX in Context to enable this preview"
                if not aod_ready
                else "Show the filled AoD filter sector at the TX selected in Context"
            )
        )

        if selection_label is not None:
            selection_label.setText(
                "AoD TX: "
                f"{self._format_node_selection(selected_tx, 'TX')}  |  "
                "AoA RX: "
                f"{self._format_node_selection(selected_rx, 'RX')}"
            )
            selection_label.setVisible(True)

        def apply_status(label, text: str, ready: bool) -> None:
            """Apply compact ready/not-ready styling to a preview status label."""
            if label is None:
                return
            color = "#2e7d32" if ready else "#8a6d3b"
            label.setText(text)
            label.setStyleSheet(f"font-size: 9px; color: {color};")

        if not renderer_ready:
            aoa_status = "AoA: pygfx only"
            aod_status = "AoD: pygfx only"
        else:
            aoa_status = "AoA: ready" if aoa_ready else "AoA: select one RX"
            aod_status = "AoD: ready" if aod_ready else "AoD: select one TX"
        if renderer_ready and aoa_ready and not aoa_cb.isChecked():
            aoa_status = "AoA: ready, enable to show"
        if renderer_ready and aod_ready and not aod_cb.isChecked():
            aod_status = "AoD: ready, enable to show"

        apply_status(aoa_status_label, aoa_status, renderer_ready and aoa_ready)
        apply_status(aod_status_label, aod_status, renderer_ready and aod_ready)
        if status_row is not None:
            status_row.setVisible(True)

        status = f"{aod_status} | {aoa_status}"
        status_label.setText(status)
        status_label.setVisible(False)

    def _on_aoa_aperture_toggled(self, checked: bool) -> None:
        """Handle AOA aperture visibility toggle."""
        from shared.logging import get_logger

        logger = get_logger("orchav.panels.mpc_panel")
        logger.info(f"_on_aoa_aperture_toggled called with checked={checked}")
        if not hasattr(self, "parent") or not self.parent:
            logger.warning("_on_aoa_aperture_toggled: no parent")
            return
        if hasattr(self.parent, "ui_controller"):
            logger.info(
                f"_on_aoa_aperture_toggled: calling ui_controller.handle_aoa_aperture_toggled({checked})"
            )
            self.parent.ui_controller.handle_aoa_aperture_toggled(checked)
        else:
            logger.warning("_on_aoa_aperture_toggled: parent has no ui_controller")
        self.refresh_aperture_preview_state()

    def _on_global_angular_reference_toggled(self, checked: bool) -> None:
        """Handle global angular reference visibility toggle."""
        if not hasattr(self, "parent") or not self.parent:
            return
        if hasattr(self.parent, "ui_controller"):
            self.parent.ui_controller.handle_global_angular_reference_toggled(checked)
        self.refresh_aperture_preview_state()

    def _on_local_angular_reference_toggled(self, checked: bool) -> None:
        """Handle local angular reference visibility toggle."""
        if not hasattr(self, "parent") or not self.parent:
            return
        if hasattr(self.parent, "ui_controller"):
            self.parent.ui_controller.handle_local_angular_reference_toggled(checked)
        self.refresh_aperture_preview_state()

    def _on_aod_aperture_toggled(self, checked: bool) -> None:
        """Handle AOD aperture visibility toggle."""
        from shared.logging import get_logger

        logger = get_logger("orchav.panels.mpc_panel")
        logger.info(f"_on_aod_aperture_toggled called with checked={checked}")
        if not hasattr(self, "parent") or not self.parent:
            logger.warning("_on_aod_aperture_toggled: no parent")
            return
        if hasattr(self.parent, "ui_controller"):
            logger.info(
                f"_on_aod_aperture_toggled: calling ui_controller.handle_aod_aperture_toggled({checked})"
            )
            self.parent.ui_controller.handle_aod_aperture_toggled(checked)
        else:
            logger.warning("_on_aod_aperture_toggled: parent has no ui_controller")
        self.refresh_aperture_preview_state()

    def _on_aperture_radius_changed(self, value: float) -> None:
        """Handle aperture radius change."""
        from shared.logging import get_logger

        logger = get_logger("orchav.panels.mpc_panel")
        logger.info(f"_on_aperture_radius_changed called with value={value}")
        if not hasattr(self, "parent") or not self.parent:
            return
        if hasattr(self.parent, "ui_controller"):
            self.parent.ui_controller.handle_aperture_radius_changed(value)
        self.refresh_aperture_preview_state()

    def update_range_filter_bounds(
        self,
        delay_min: float = 0.0,
        delay_max: float = 10000.0,
        loss_min: float = 0.0,
        loss_max: float = 200.0,
        aoa_az_min: float = -180.0,
        aoa_az_max: float = 180.0,
        aoa_el_min: float = -90.0,
        aoa_el_max: float = 90.0,
        aod_az_min: float = -180.0,
        aod_az_max: float = 180.0,
        aod_el_min: float = -90.0,
        aod_el_max: float = 90.0,
    ) -> None:
        """Update the spin box ranges based on canonical data ranges.

        This is called when a new frame is loaded to update the UI with
        the actual data ranges from the current frame.

        Args:
            delay_min: Minimum delay value in the data (ns)
            delay_max: Maximum delay value in the data (ns)
            loss_min: Minimum path loss value in the data (dB)
            loss_max: Maximum path loss value in the data (dB)
            aoa_az_min: Minimum AoA azimuth (degrees)
            aoa_az_max: Maximum AoA azimuth (degrees)
            aoa_el_min: Minimum AoA elevation (degrees)
            aoa_el_max: Maximum AoA elevation (degrees)
            aod_az_min: Minimum AoD azimuth (degrees)
            aod_az_max: Maximum AoD azimuth (degrees)
            aod_el_min: Minimum AoD elevation (degrees)
            aod_el_max: Maximum AoD elevation (degrees)
        """
        # Block signals while updating ranges to avoid triggering filter changes
        for widget_name in [
            "delay_filter_min",
            "delay_filter_max",
            "power_filter_min",
            "power_filter_max",
            "aoa_az_filter_min",
            "aoa_az_filter_max",
            "aoa_el_filter_min",
            "aoa_el_filter_max",
            "aod_az_filter_min",
            "aod_az_filter_max",
            "aod_el_filter_min",
            "aod_el_filter_max",
        ]:
            if widget_name in self.widgets:
                self.widgets[widget_name].blockSignals(True)

        # Add small padding to ranges to avoid edge cases
        delay_pad = max(0.1, (delay_max - delay_min) * 0.01)
        loss_pad = max(0.1, (loss_max - loss_min) * 0.01)

        if "delay_filter_min" in self.widgets:
            self.widgets["delay_filter_min"].setRange(delay_min - delay_pad, delay_max + delay_pad)
            self.widgets["delay_filter_min"].setValue(delay_min - delay_pad)
        if "delay_filter_max" in self.widgets:
            self.widgets["delay_filter_max"].setRange(delay_min - delay_pad, delay_max + delay_pad)
            self.widgets["delay_filter_max"].setValue(delay_max + delay_pad)

        if "power_filter_min" in self.widgets:
            self.widgets["power_filter_min"].setRange(loss_min - loss_pad, loss_max + loss_pad)
            self.widgets["power_filter_min"].setValue(loss_min - loss_pad)
        if "power_filter_max" in self.widgets:
            self.widgets["power_filter_max"].setRange(loss_min - loss_pad, loss_max + loss_pad)
            self.widgets["power_filter_max"].setValue(loss_max + loss_pad)

        # Angle filters use normalized angular coordinates, independent of
        # the storage convention used by the frame data (0..360 or -180..180).
        # Keep the widget bounds natural so values such as -20..20 remain
        # selectable even when the current frame stores equivalent azimuths
        # around 340..360 degrees.
        az_widget_min, az_widget_max = -180.0, 180.0
        el_widget_min, el_widget_max = -90.0, 90.0

        if "aoa_az_filter_min" in self.widgets:
            self.widgets["aoa_az_filter_min"].setRange(az_widget_min, az_widget_max)
            self.widgets["aoa_az_filter_min"].setValue(az_widget_min)
        if "aoa_az_filter_max" in self.widgets:
            self.widgets["aoa_az_filter_max"].setRange(az_widget_min, az_widget_max)
            self.widgets["aoa_az_filter_max"].setValue(az_widget_max)

        if "aoa_el_filter_min" in self.widgets:
            self.widgets["aoa_el_filter_min"].setRange(el_widget_min, el_widget_max)
            self.widgets["aoa_el_filter_min"].setValue(el_widget_min)
        if "aoa_el_filter_max" in self.widgets:
            self.widgets["aoa_el_filter_max"].setRange(el_widget_min, el_widget_max)
            self.widgets["aoa_el_filter_max"].setValue(el_widget_max)

        if "aod_az_filter_min" in self.widgets:
            self.widgets["aod_az_filter_min"].setRange(az_widget_min, az_widget_max)
            self.widgets["aod_az_filter_min"].setValue(az_widget_min)
        if "aod_az_filter_max" in self.widgets:
            self.widgets["aod_az_filter_max"].setRange(az_widget_min, az_widget_max)
            self.widgets["aod_az_filter_max"].setValue(az_widget_max)

        if "aod_el_filter_min" in self.widgets:
            self.widgets["aod_el_filter_min"].setRange(el_widget_min, el_widget_max)
            self.widgets["aod_el_filter_min"].setValue(el_widget_min)
        if "aod_el_filter_max" in self.widgets:
            self.widgets["aod_el_filter_max"].setRange(el_widget_min, el_widget_max)
            self.widgets["aod_el_filter_max"].setValue(el_widget_max)

        # Keep angle controls visible. In normal generated ORCHAV frames,
        # AoA/AoD metrics are part of the MPC data, and users expect these
        # filters to remain available across scenarios.
        if "angle_filters_container" in self.widgets:
            self.widgets["angle_filters_container"].setVisible(True)

        # Re-enable signals
        for widget_name in [
            "delay_filter_min",
            "delay_filter_max",
            "power_filter_min",
            "power_filter_max",
            "aoa_az_filter_min",
            "aoa_az_filter_max",
            "aoa_el_filter_min",
            "aoa_el_filter_max",
            "aod_az_filter_min",
            "aod_az_filter_max",
            "aod_el_filter_min",
            "aod_el_filter_max",
        ]:
            if widget_name in self.widgets:
                self.widgets[widget_name].blockSignals(False)

        self.refresh_aperture_preview_state()

    def _create_preset_controls(self) -> QWidget:
        """Create preset dropdown and save button.

        Presets are applied immediately when selected (no Load button needed).
        The reset preset shows all MPCs.
        """
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        # Label
        preset_label = QLabel("Preset:")
        preset_label.setStyleSheet("font-weight: bold; font-size: 10px;")
        layout.addWidget(preset_label)

        # Preset dropdown - presets apply immediately on selection
        self.widgets["preset_combo"] = QComboBox()
        self.widgets["preset_combo"].setStyleSheet("QComboBox { font-size: 10px; }")
        self.widgets["preset_combo"].setToolTip(
            "Select a filter preset - applies immediately.\n"
            "Use the reset preset to show all MPCs."
        )
        # Connect to immediately apply preset when selection changes
        self.widgets["preset_combo"].currentTextChanged.connect(self._on_preset_selected)
        layout.addWidget(self.widgets["preset_combo"], 1)

        self.widgets["save_preset_btn"] = QPushButton("Save...")
        self.widgets["save_preset_btn"].setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 4px 10px;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:pressed {
                background-color: #229954;
            }
            """)
        self.widgets["save_preset_btn"].setToolTip("Save current filters as a new preset")
        self.widgets["save_preset_btn"].clicked.connect(self._on_save_preset)
        layout.addWidget(self.widgets["save_preset_btn"])

        return container

    def _create_separator(self) -> QFrame:
        """Create the thin separator used inside dense MPC filter groups."""
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("color: #ddd; max-height: 1px;")
        return separator

    def set_reconstruction_type_visible(self, visible: bool):
        """Show or hide the reconstruction_type color mode based on data source."""
        rb = self.widgets.get("reconstruction_type_rb")
        if rb:
            rb.setVisible(visible)
            if not visible and rb.isChecked():
                # If reconstruction_type is currently selected but being hidden,
                # switch to reflection_order
                reflection_rb = self.widgets.get("reflection_order_rb")
                if reflection_rb:
                    reflection_rb.setChecked(True)

    def _create_material_bulk_actions(self) -> QHBoxLayout:
        """Create bulk action buttons for material selection."""
        bulk_layout = QHBoxLayout()
        bulk_layout.setSpacing(4)
        bulk_layout.setContentsMargins(0, 2, 0, 2)

        # "All" button
        select_all_btn = QPushButton("All")
        select_all_btn.setToolTip("Select all materials")
        select_all_btn.setStyleSheet("""
            QPushButton {
                font-size: 9px;
                padding: 2px 8px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            """)
        select_all_btn.clicked.connect(self._on_select_all_materials)
        bulk_layout.addWidget(select_all_btn)

        # "None" button
        deselect_all_btn = QPushButton("None")
        deselect_all_btn.setToolTip("Deselect all materials")
        deselect_all_btn.setStyleSheet("""
            QPushButton {
                font-size: 9px;
                padding: 2px 8px;
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:pressed {
                background-color: #c4150a;
            }
            """)
        deselect_all_btn.clicked.connect(self._on_deselect_all_materials)
        bulk_layout.addWidget(deselect_all_btn)

        # "Invert" button
        invert_btn = QPushButton("Invert")
        invert_btn.setToolTip("Invert material selection")
        invert_btn.setStyleSheet("""
            QPushButton {
                font-size: 9px;
                padding: 2px 8px;
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:pressed {
                background-color: #0a6bc2;
            }
            """)
        invert_btn.clicked.connect(self._on_invert_material_selection)
        bulk_layout.addWidget(invert_btn)

        bulk_layout.addStretch()
        return bulk_layout

    def _on_select_all_materials(self):
        """Select all material rows."""
        self._set_all_material_rows(Qt.Checked)
        logger.debug("Selected all materials")

    def _on_deselect_all_materials(self):
        """Deselect all material rows."""
        self._set_all_material_rows(Qt.Unchecked)
        logger.debug("Deselected all materials")

    def _on_invert_material_selection(self):
        """Invert all material row states."""
        model = self.widgets.get("materials_model")
        if model is None:
            return
        for row in range(model.rowCount()):
            item = model.item(row)
            if item is None or not item.isCheckable() or not item.isEnabled():
                continue
            item.setCheckState(Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked)
        logger.debug("Inverted material selection")

    def _set_all_material_rows(self, state: Qt.CheckState) -> None:
        """Set every enabled material row to *state*."""
        model = self.widgets.get("materials_model")
        if model is None:
            return
        for row in range(model.rowCount()):
            item = model.item(row)
            if item is not None and item.isCheckable() and item.isEnabled():
                item.setCheckState(state)

    def material_filter_active(self) -> bool:
        """Return whether any material row is currently excluded."""
        model = self.widgets.get("materials_model")
        if model is None:
            return False
        for row in range(model.rowCount()):
            item = model.item(row)
            if item is not None and item.isCheckable() and item.isEnabled():
                if item.checkState() != Qt.Checked:
                    return True
        return False

    def _on_preset_selected(self, preset_name: str):
        """Apply the selected preset immediately.

        This is called whenever the preset dropdown selection changes.
        The preset filters are applied immediately without needing a Load button.
        """
        if not preset_name:
            return

        if not hasattr(self.parent, "preset_service"):
            logger.error("PresetService not available on parent visualizer")
            return

        preset_data = self.parent.preset_service.load_preset(preset_name)
        if not preset_data:
            logger.warning(f"Failed to load preset: {preset_name}")
            return

        logger.info(f"Applying preset: {preset_name}")

        viz = self.parent

        # Update checkbox UI (block signals to avoid individual updates)
        new_orders = set(viz.app_state.mpc_allowed_orders)
        if "mpc_allowed_orders" in preset_data:
            new_orders = set(preset_data["mpc_allowed_orders"])
            for i in MPC_ORDER_VALUES:
                checkbox = self.widgets.get(f"order_{i}_cb")
                if checkbox:
                    checkbox.blockSignals(True)
                    checkbox.setChecked(i in new_orders)
                    checkbox.blockSignals(False)

        new_types = set(viz.app_state.mpc_allowed_types)
        if "mpc_allowed_types" in preset_data:
            new_types = set(preset_data["mpc_allowed_types"])
            for value in MPC_TYPE_VALUES:
                checkbox = self.widgets.get(f"type_{value}_cb")
                if checkbox:
                    checkbox.blockSignals(True)
                    checkbox.setChecked(value in new_types)
                    checkbox.blockSignals(False)

        delay_min = preset_data.get("delay_filter_min_ns", None)
        delay_max = preset_data.get("delay_filter_max_ns", None)
        power_min = preset_data.get("power_filter_min_db", None)
        power_max = preset_data.get("power_filter_max_db", None)
        aoa_az_min = preset_data.get("aoa_az_filter_min_deg", None)
        aoa_az_max = preset_data.get("aoa_az_filter_max_deg", None)
        aoa_el_min = preset_data.get("aoa_el_filter_min_deg", None)
        aoa_el_max = preset_data.get("aoa_el_filter_max_deg", None)
        aod_az_min = preset_data.get("aod_az_filter_min_deg", None)
        aod_az_max = preset_data.get("aod_az_filter_max_deg", None)
        aod_el_min = preset_data.get("aod_el_filter_min_deg", None)
        aod_el_max = preset_data.get("aod_el_filter_max_deg", None)

        range_filter_widgets = [
            ("delay_filter_min", delay_min),
            ("delay_filter_max", delay_max),
            ("power_filter_min", power_min),
            ("power_filter_max", power_max),
            ("aoa_az_filter_min", aoa_az_min),
            ("aoa_az_filter_max", aoa_az_max),
            ("aoa_el_filter_min", aoa_el_min),
            ("aoa_el_filter_max", aoa_el_max),
            ("aod_az_filter_min", aod_az_min),
            ("aod_az_filter_max", aod_az_max),
            ("aod_el_filter_min", aod_el_min),
            ("aod_el_filter_max", aod_el_max),
        ]
        for widget_name, value in range_filter_widgets:
            if widget_name in self.widgets:
                self.widgets[widget_name].blockSignals(True)
                if value is None:
                    if "min" in widget_name:
                        self.widgets[widget_name].setValue(self.widgets[widget_name].minimum())
                    else:
                        self.widgets[widget_name].setValue(self.widgets[widget_name].maximum())
                else:
                    self.widgets[widget_name].setValue(value)
                self.widgets[widget_name].blockSignals(False)

        viz.set_state(
            mpc_allowed_orders=frozenset(new_orders),
            mpc_allowed_types=frozenset(new_types),
            # Range filters
            delay_filter_min_ns=delay_min,
            delay_filter_max_ns=delay_max,
            power_filter_min_db=power_min,
            power_filter_max_db=power_max,
            aoa_az_filter_min_deg=aoa_az_min,
            aoa_az_filter_max_deg=aoa_az_max,
            aoa_el_filter_min_deg=aoa_el_min,
            aoa_el_filter_max_deg=aoa_el_max,
            aod_az_filter_min_deg=aod_az_min,
            aod_az_filter_max_deg=aod_az_max,
            aod_el_filter_min_deg=aod_el_min,
            aod_el_filter_max_deg=aod_el_max,
        )

        invalidate_visualizer_cache(viz, CacheInvalidationScope.FILTERS, reason="mpc_filter_preset")
        if hasattr(viz, "schedule_update"):
            viz.schedule_update()

        logger.info(f"Preset '{preset_name}' applied successfully")

    def _on_save_preset(self):
        """Save current filter state as a new preset."""
        # Prompt for preset name
        preset_name, ok = QInputDialog.getText(
            self.parent,
            "Save Preset",
            "Enter preset name:",
        )

        if not ok or not preset_name:
            return

        if not hasattr(self.parent, "preset_service"):
            logger.error("PresetService not available on parent visualizer")
            return

        # Collect current filter state
        filter_state = {
            "mpc_allowed_orders": [],
            "mpc_allowed_types": [],
            "description": f"User-defined preset: {preset_name}",
        }

        for i in MPC_ORDER_VALUES:
            checkbox = self.widgets.get(f"order_{i}_cb")
            if checkbox and checkbox.isChecked():
                filter_state["mpc_allowed_orders"].append(i)

        for value in MPC_TYPE_VALUES:
            checkbox = self.widgets.get(f"type_{value}_cb")
            if checkbox and checkbox.isChecked():
                filter_state["mpc_allowed_types"].append(value)

        success = self.parent.preset_service.save_preset(preset_name, filter_state)
        if success:
            logger.info(f"Preset '{preset_name}' saved successfully")
            # Refresh preset dropdown
            self.refresh_preset_list()
        else:
            logger.error(f"Failed to save preset: {preset_name}")

    def refresh_preset_list(self):
        """Refresh the preset dropdown with available presets.

        The first preset resets filters and is selected by default.
        Presets are applied immediately when selected.
        """
        preset_combo = self.widgets.get("preset_combo")
        # Note: Must use 'is None' check because empty QComboBox is falsy!
        if preset_combo is None:
            logger.warning("refresh_preset_list: preset_combo widget not found")
            return

        if not hasattr(self.parent, "preset_service"):
            logger.debug("refresh_preset_list: parent has no preset_service")
            return

        # Save current selection (if any)
        current_selection = preset_combo.currentText()

        with QSignalBlocker(preset_combo):
            # Clear and repopulate without a placeholder; the first preset resets filters.
            preset_combo.clear()

            presets = self.parent.preset_service.list_presets()
            for preset_name in presets:
                preset_combo.addItem(preset_name)

            # Restore selection if it still exists, otherwise keep first item selected
            if current_selection:
                index = preset_combo.findText(current_selection)
                if index >= 0:
                    preset_combo.setCurrentIndex(index)

        logger.debug(f"Refreshed preset list: {len(presets)} presets available")

    def set_materials(self, material_ids: list[str], checked: set[str] | None):
        """Populate material filter rows from detected MPC materials."""
        model: QStandardItemModel | None = self.widgets.get("materials_model")
        if model is None:
            logger.warning("Materials model not found in widgets.")
            return

        self._updating_material_model = True
        try:
            model.clear()
            if not material_ids:
                item = QStandardItem("(No materials detected)")
                item.setEnabled(False)
                item.setEditable(False)
                model.appendRow(item)
            else:
                checked_materials = set(checked or set())
                for material_id in sorted(material_ids):
                    item = QStandardItem(material_id)
                    item.setEditable(False)
                    item.setCheckable(True)
                    item.setCheckState(
                        Qt.Checked if material_id in checked_materials else Qt.Unchecked
                    )
                    item.setData(material_id, Qt.UserRole)
                    model.appendRow(item)
        finally:
            self._updating_material_model = False

        view: QListView | None = self.widgets.get("materials_view")
        if view is not None:
            view.doItemsLayout()
            view.updateGeometry()
            view.viewport().update()

        ui_mgr = getattr(self.parent, "ui_manager", None)
        if ui_mgr is not None and hasattr(ui_mgr, "update_paths_tab_badge"):
            ui_mgr.update_paths_tab_badge()

    def _on_material_item_changed(self, item: QStandardItem) -> None:
        """Forward check-state changes from the model to the MPC controller."""
        if self._updating_material_model or item is None or not item.isCheckable():
            return
        material_id = item.data(Qt.UserRole)
        if not material_id:
            return

        state = item.checkState()
        controller = getattr(self.parent, "ui_controller", None)
        handler = (
            getattr(controller, "handle_mpc_material_filter_changed", None)
            if controller
            else getattr(self.parent, "on_mpc_material_filter_changed", None)
        )
        if handler:
            handler(material_id, state.value)

        ui_mgr = getattr(self.parent, "ui_manager", None)
        badge_handler = (
            getattr(ui_mgr, "update_paths_tab_badge", None) if ui_mgr is not None else None
        )
        if badge_handler is not None:
            badge_handler()

    def _create_color_legend_group(self) -> QWidget:
        """Create the compact inline color legend used by MPC color modes."""
        group = QWidget()
        self.widgets["color_legend_layout"] = QVBoxLayout(group)
        self.widgets["color_legend_layout"].setSpacing(2)
        self.widgets["color_legend_layout"].setContentsMargins(0, 0, 0, 0)

        # Single legend row: [optional colorbar] [swatch label] [swatch label] ...
        legend_row = QWidget()
        legend_row_layout = QHBoxLayout(legend_row)
        legend_row_layout.setSpacing(6)
        legend_row_layout.setContentsMargins(0, 0, 0, 0)

        self.widgets["colorbar_widget"] = QWidget()
        self.widgets["colorbar_widget"].setFixedHeight(14)
        self.widgets["colorbar_widget"].setFixedWidth(100)
        self.widgets["colorbar_widget"].setStyleSheet(
            self._colorbar_style(f"background-color: {current_theme().bg_tertiary};")
        )
        self.widgets["colorbar_widget"].hide()
        legend_row_layout.addWidget(self.widgets["colorbar_widget"])

        # Container for inline legend items
        self.widgets["legend_items_container"] = QWidget()
        self.widgets["legend_items_layout"] = QHBoxLayout(self.widgets["legend_items_container"])
        self.widgets["legend_items_layout"].setSpacing(6)
        self.widgets["legend_items_layout"].setContentsMargins(0, 0, 0, 0)
        legend_row_layout.addWidget(self.widgets["legend_items_container"])
        legend_row_layout.addStretch()

        self.widgets["color_legend_layout"].addWidget(legend_row)

        return group

    def _clear_layout(self, layout) -> None:
        """Remove all items from a layout."""
        while layout.count() > 0:
            item = layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

    def update_color_legend(self, color_mode: str = "reflection_order") -> None:
        """Refresh legend items for the active MPC color mode."""
        self._current_color_mode = color_mode

        if "legend_items_layout" not in self.widgets:
            return
        if "colorbar_widget" not in self.widgets or self.widgets["colorbar_widget"] is None:
            logger.warning("Colorbar widget not available for legend update")
            return

        self._clear_layout(self.widgets["legend_items_layout"])

        use_colorbar = False
        legend_tooltips = {}
        legend_items = []

        if color_mode == "reflection_order":
            order_colors = _get_order_palette_colors()
            legend_items = [
                ("LoS", order_colors[0]),
                ("1st", order_colors[1]),
                ("2nd", order_colors[2]),
                ("3rd", order_colors[3]),
                ("4th", order_colors[4]),
                ("5th", order_colors[5]),
                ("6+", order_colors[6]),
            ]
            legend_tooltips = {
                "LoS": "Direct path without any surface interaction.",
                "1st": "Paths with exactly one surface bounce.",
                "2nd": "Paths that include two reflections.",
                "3rd": "Paths reflecting three times before arrival.",
                "4th": "Paths with four surface interactions.",
                "5th": "Paths with five surface interactions.",
                "6+": "Highly scattered paths with six or more bounces.",
            }
        elif color_mode == "mpc_type":
            type_entries = _get_type_legend_entries(self._present_mpc_type_codes)
            legend_items = [(entry.label, rgb_to_css_hex(entry.color)) for entry in type_entries]
            legend_tooltips = {entry.label: entry.tooltip for entry in type_entries}
        elif color_mode == "delay":
            use_colorbar = True
            cmap_colors = _get_colormap_colors_at_positions([0.0, 0.5, 1.0])
            legend_items = [
                ("Low", cmap_colors[0]),
                ("Med", cmap_colors[1]),
                ("High", cmap_colors[2]),
            ]
            legend_tooltips = {
                "Low": "Early-arriving multipath components.",
                "Med": "Mid-range arrival times.",
                "High": "Late arrivals with long propagation paths.",
            }
            gradient_css = _get_colormap_gradient_css()
            self.widgets["colorbar_widget"].setStyleSheet(
                self._colorbar_style(f"background: {gradient_css};")
            )
        elif color_mode == "path_loss":
            use_colorbar = True
            cmap_colors = _get_colormap_colors_at_positions([0.0, 0.5, 1.0])
            legend_items = [
                ("Low", cmap_colors[0]),
                ("Med", cmap_colors[1]),
                ("High", cmap_colors[2]),
            ]
            legend_tooltips = {
                "Low": "Strong paths with minimal attenuation.",
                "Med": "Moderate attenuation along the path.",
                "High": "Weak paths with heavy attenuation.",
            }
            gradient_css = _get_colormap_gradient_css()
            self.widgets["colorbar_widget"].setStyleSheet(
                self._colorbar_style(f"background: {gradient_css};")
            )
        elif color_mode == "material":
            legend_items = self._get_material_legend_items()
        elif color_mode == "reconstruction_type":
            legend_items = [
                ("LoS", "#4D80E6"),
                ("Resolved", "#33CC4D"),
                ("Virtual", "#FF9933"),
                ("Unknown", "#808080"),
            ]
            legend_tooltips = {
                "LoS": "Direct Line-of-Sight path.",
                "Resolved": "Path fully resolved using scene geometry.",
                "Virtual": "Path reconstructed using virtual bounce points.",
                "Unknown": "Interaction type not recognized.",
            }

        # Show/hide colorbar
        self.widgets["colorbar_widget"].setVisible(use_colorbar)

        # Populate legend items inline
        target_layout = self.widgets["legend_items_layout"]
        if color_mode == "mpc_type" and not legend_items:
            empty_label = QLabel("No visible path types")
            empty_label.setObjectName("mpcTypeLegendEmptyLabel")
            empty_label.setStyleSheet(
                f"font-size: 9px; color: {current_theme().text_secondary}; font-style: italic;"
            )
            target_layout.addWidget(empty_label)
            return
        for description, hex_color in legend_items:
            tooltip_text = legend_tooltips.get(description)
            if tooltip_text is None and color_mode == "material":
                tooltip_text = f"Material: {description}"
            target_layout.addWidget(self._create_legend_item(description, hex_color, tooltip_text))

    def set_present_mpc_type_codes(self, type_codes: tuple[int, ...]) -> None:
        """Adopt the interaction codes from one renderer-accepted frame."""
        normalized = tuple(sorted({int(value) for value in type_codes}))
        if normalized == self._present_mpc_type_codes:
            return
        self._present_mpc_type_codes = normalized
        if self._current_color_mode == "mpc_type":
            self.update_color_legend("mpc_type")

    def _create_legend_item(self, description: str, hex_color: str, tooltip: str | None) -> QWidget:
        """Create a text-safe color legend item with a painted Qt swatch."""
        theme = current_theme()
        item = QWidget()
        layout = QHBoxLayout(item)
        layout.setSpacing(3)
        layout.setContentsMargins(0, 0, 0, 0)

        swatch = QFrame()
        swatch.setFixedSize(9, 9)
        swatch.setStyleSheet(
            f"background-color: {hex_color}; "
            f"border: 1px solid {theme.border_primary}; "
            "border-radius: 4px;"
        )
        layout.addWidget(swatch)

        label = QLabel(description)
        label.setStyleSheet(f"font-size: 9px; color: {theme.text_primary}; font-weight: bold;")
        layout.addWidget(label)

        if tooltip:
            item.setToolTip(tooltip)
            swatch.setToolTip(tooltip)
            label.setToolTip(tooltip)
        return item

    @staticmethod
    def _colorbar_style(background_rule: str) -> str:
        """Return theme-aware colorbar frame styling."""
        theme = current_theme()
        return f"{background_rule} border: 1px solid {theme.border_primary}; border-radius: 3px;"

    def _get_material_legend_items(self):
        """Return material legend rows from the MPC core color policy."""
        legend_items = []
        if hasattr(self.parent, "mpc_core") and self.parent.mpc_core is not None:
            use_distinct = getattr(self.parent, "app_state", None)
            use_distinct = use_distinct.use_distinct_material_colors if use_distinct else False
            material_items = self.parent.mpc_core.material_legend_items(use_distinct)
            if material_items:
                for material_id, color in material_items:
                    hex_color = (
                        f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
                    )
                    short_name = material_id.replace("mat-itu_", "").replace("mat-", "")
                    legend_items.append((short_name, hex_color))
            else:
                legend_items.append(("No materials", current_theme().warning))
        else:
            legend_items.append(("No MPC core", current_theme().warning))
        return legend_items
