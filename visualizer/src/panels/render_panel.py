"""Rendering, lighting, and figure-capture controls for the visualizer shell.

``RenderPanel`` builds renderer-neutral controls first, then uses renderer
capabilities to reveal backend-specific controls such as pygfx clipping planes,
pygfx light rigs, or Open3D's native settings panel. Handlers update app state,
renderer protocol hooks, or scene-appearance services without owning renderer
implementation policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from shared.logging import get_logger
from shared.scenarios.app_paths import load_app_paths
from shared.scenarios.paths import find_project_root

from ..lighting.assets import environment_catalog
from ..renderers.protocol import renderer_capabilities
from ..scene.defaults import DEFAULT_SCENE_BACKGROUND_PRESET, SCENE_BACKGROUND_PRESETS
from ..state import normalize_viewport_hud_mode, normalize_viewport_hud_style
from .base import BasePanel
from .collapsible_section import CollapsibleSection
from .environment_download import EnvironmentDownload
from .render_effects import (
    create_contour_controls,
    create_dusk_controls,
    sync_effect_controls,
)
from .scene_guide_controls import (
    create_dimensions_control,
    create_grid_compass_controls,
    restore_guide_controls,
    sync_guide_controls,
)

logger = get_logger("orchav.render_panel")

# Debounce delay in milliseconds for slider updates
SLIDER_DEBOUNCE_MS = 150


class RenderPanel(BasePanel):
    """Create rendering widgets and synchronize them with renderer capabilities."""

    _BG_PRESETS = SCENE_BACKGROUND_PRESETS
    _HOVER_INFO_MODES = (
        ("Off", "off"),
        ("Essential", "essential"),
        ("Inspect All", "inspect_all"),
    )
    _HOVER_INFO_LABEL_BY_MODE = {
        "off": "Off",
        "essential": "Essential",
        "inspect_all": "Inspect All",
    }

    # Paper mode uses one capability-filtered preset across renderers.
    _PAPER_MODE_PRESET = {
        "bg_combo": "White",
        "shader_combo": "Unlit",
        "skybox_cb": False,
        "ibl_intensity": 5000,
        "shadows_cb": False,
        "outline_cb": True,
    }

    def __init__(self, parent=None):
        """Initialize renderer-specific widget registries and IBL library state."""
        # Debounce timers for opacity sliders avoid repainting on every drag tick.
        self._building_alpha_timer = None
        self._target_alpha_timer = None
        self._pending_building_alpha = None
        self._pending_target_alpha = None
        # Paper mode: saved widget values before activation
        self._paper_mode_saved: Optional[Dict] = None
        # Baseline renderer-control containers shown whenever a renderer exists.
        self._renderer_control_widgets: list[QWidget] = []
        self._open3d_scene_shader_widgets: list[QWidget] = []
        self._clipping_widgets: list[QWidget] = []
        self._quality_widgets: list[QWidget] = []
        self._viewport_hud_widgets: list[QWidget] = []
        self._pygfx_renderer_widgets: list[QWidget] = []
        self._syncing_lighting_profile = False
        self._background_skybox_shown: bool | None = None
        # IBL library (name -> identifier/path)
        self._ibl_library = {"Default": "default"}
        self._ibl_custom_index = 0
        self._ibl_dir: Optional[Path] = None
        self._hdr_ibl_mode = renderer_capabilities(getattr(parent, "renderer", None)).hdr_ibl
        self._project_root = find_project_root(Path.cwd())
        self._pygfx_light_widgets: list[QWidget] = []
        self._environment_light_widgets: list[QWidget] = []
        self._advanced_light_widgets: list[QWidget] = []
        self._load_ibl_library_from_disk()
        super().__init__(parent)

    def _make_row_container(self, row_layout: QHBoxLayout) -> QWidget:
        """Wrap a QHBoxLayout in a QWidget container for show/hide toggling."""
        container = QWidget()
        container.setLayout(row_layout)
        row_layout.setContentsMargins(0, 0, 0, 0)
        return container

    def create_panel(self) -> QGroupBox:
        """Create combined panel (kept for backward compatibility)."""
        group = self.create_group_box("Render")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.create_scene_style_panel())
        layout.addWidget(self.create_scene_view_panel())
        layout.addWidget(self.create_lighting_panel())
        layout.addWidget(self.create_image_quality_panel())
        layout.addWidget(self.create_viewport_hud_panel())
        layout.addStretch()
        QTimer.singleShot(0, self._sync_from_visualizer)
        return group

    # Sub-panel: Appearance / style / view

    def create_appearance_panel(self) -> QWidget:
        """Create combined appearance controls."""
        group = self.create_group_box("Appearance")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.create_scene_style_panel())
        layout.addWidget(self.create_scene_view_panel())
        layout.addWidget(self.create_image_quality_panel())
        layout.addStretch()
        return group

    def create_scene_style_panel(self) -> QWidget:
        """Create the primary scene styling controls."""
        group = self.create_group_box("Surfaces")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        self.widgets["paper_mode_cb"] = QCheckBox("Paper Mode")
        self.widgets["paper_mode_cb"].setChecked(False)
        self.widgets["paper_mode_cb"].setToolTip(
            "One-click preset for publication-ready views:\n"
            "white background, flat lighting, shadows off,\n"
            "and building outlines when supported."
        )
        self.widgets["paper_mode_cb"].toggled.connect(self._on_paper_mode_toggled)
        layout.addWidget(self.widgets["paper_mode_cb"])
        if getattr(self.parent, "appearance_preset_service", None) is not None:
            self.widgets["paper_mode_cb"].hide()

        self.widgets["texture_mipmaps_cb"] = QCheckBox("Texture mipmaps")
        self.widgets["texture_mipmaps_cb"].setToolTip(
            "Less shimmer, softer distant detail. Uses prefiltered texture levels, "
            "with about one third more texture memory. Turn off for sharper "
            "facades with more potential shimmer. Enable Texture maps to use this control. "
            "Changes surface textures only, not geometry or volume data."
        )
        self.widgets["texture_mipmaps_cb"].toggled.connect(self._on_texture_mipmaps_toggled)
        layout.addWidget(self.widgets["texture_mipmaps_cb"])

        # Opacity sliders use renderer alpha semantics: 0 is invisible, 100 is solid.
        opacity_group = self.create_subgroup_box("Opacity")
        opacity_layout = QVBoxLayout(opacity_group)
        opacity_layout.setSpacing(4)
        opacity_layout.setContentsMargins(6, 6, 6, 6)
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Scene:"))
        self.widgets["building_alpha_slider"] = QSlider(Qt.Horizontal)
        self.widgets["building_alpha_slider"].setRange(0, 100)
        self.widgets["building_alpha_slider"].setValue(100)
        self.widgets["building_alpha_slider"].setToolTip(
            "Building/scene mesh opacity (0=invisible, 100=solid)"
        )
        self.widgets["building_alpha_slider"].valueChanged.connect(
            self._on_building_alpha_slider_changed
        )
        opacity_row.addWidget(self.widgets["building_alpha_slider"])

        opacity_layout.addLayout(opacity_row)
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Targets:"))
        self.widgets["target_alpha_slider"] = QSlider(Qt.Horizontal)
        self.widgets["target_alpha_slider"].setRange(0, 100)
        self.widgets["target_alpha_slider"].setValue(100)
        self.widgets["target_alpha_slider"].setToolTip(
            "Target mesh opacity (0=invisible, 100=solid)"
        )
        self.widgets["target_alpha_slider"].valueChanged.connect(
            self._on_target_alpha_slider_changed
        )
        opacity_row.addWidget(self.widgets["target_alpha_slider"])

        opacity_layout.addLayout(opacity_row)
        layout.addWidget(opacity_group)
        self._renderer_control_widgets.append(opacity_group)

        edges_group = self.create_subgroup_box("Edges")
        edges_layout = QVBoxLayout(edges_group)
        edges_layout.setSpacing(4)

        self.widgets["outline_cb"] = QCheckBox("Show Edges")
        self.widgets["outline_cb"].setChecked(False)
        self.widgets["outline_cb"].setToolTip("Show wireframe edges on all scene meshes")
        self.widgets["outline_cb"].toggled.connect(self._on_outline_toggled)
        edges_row = QHBoxLayout()
        edges_row.addWidget(self.widgets["outline_cb"])

        self.widgets["target_outline_cb"] = QCheckBox("Target Edges")
        self.widgets["target_outline_cb"].setChecked(False)
        self.widgets["target_outline_cb"].setToolTip("Show wireframe edges on target meshes")
        self.widgets["target_outline_cb"].toggled.connect(self._on_target_outline_toggled)
        edges_row.addWidget(self.widgets["target_outline_cb"])

        # Wireframe overlay (pygfx only — material-level toggle, zero geometry overhead)
        self.widgets["wireframe_cb"] = QCheckBox("Wireframe")
        self.widgets["wireframe_cb"].setChecked(False)
        self.widgets["wireframe_cb"].setToolTip("Render scene meshes as wireframe (pygfx only)")
        self.widgets["wireframe_cb"].toggled.connect(self._on_wireframe_toggled)
        edges_row.addWidget(self.widgets["wireframe_cb"])
        edges_row.addStretch()
        edges_layout.addLayout(edges_row)
        edges_layout.addWidget(self._create_edge_width_controls())
        self._pygfx_renderer_widgets.append(self.widgets["wireframe_cb"])

        layout.addWidget(edges_group)
        return group

    def create_scene_view_panel(self) -> QWidget:
        """Create controls for view helpers and scene inspection tools."""
        group = self.create_group_box("Guides and inspection")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        view_row = QHBoxLayout()
        self.widgets["show_axes_cb"] = QCheckBox("World axes (X/Y/Z)")
        self.widgets["show_axes_cb"].setChecked(False)
        self.widgets["show_axes_cb"].setToolTip("Show XYZ coordinate axes at origin")
        self.widgets["show_axes_cb"].toggled.connect(self._on_show_axes_toggled)
        view_row.addWidget(self.widgets["show_axes_cb"])

        self.widgets["screen_space_labels_cb"] = QCheckBox("Screen-space labels")
        self.widgets["screen_space_labels_cb"].setChecked(True)
        self.widgets["screen_space_labels_cb"].setToolTip(
            "Render TX/RX labels at a fixed pixel size regardless of zoom (pygfx only)"
        )
        self.widgets["screen_space_labels_cb"].toggled.connect(self._on_screen_space_labels_toggled)

        view_row.addStretch()
        view_container = self._make_row_container(view_row)
        layout.addWidget(view_container)
        self._renderer_control_widgets.append(view_container)
        layout.addWidget(create_grid_compass_controls(self))
        layout.addWidget(create_dimensions_control(self))

        hover_row = QHBoxLayout()
        hover_row.addWidget(QLabel("Hover Info:"))
        self.widgets["hover_info_combo"] = QComboBox()
        for label, mode in self._HOVER_INFO_MODES:
            self.widgets["hover_info_combo"].addItem(label, mode)
        self.widgets["hover_info_combo"].setCurrentText("Essential")
        self.widgets["hover_info_combo"].setToolTip(
            "Off: no hover labels. Essential: rays, TX/RX, and targets only. "
            "Inspect All: also show scene geometry. "
            "Requires the HUD switch and Rendering > Viewport HUD > Hover to be enabled."
        )
        self.widgets["hover_info_combo"].currentIndexChanged.connect(
            self._on_hover_info_mode_changed
        )
        hover_row.addWidget(self.widgets["hover_info_combo"])
        hover_row.addWidget(self.widgets["screen_space_labels_cb"])
        hover_row.addStretch()
        hover_container = self._make_row_container(hover_row)
        self.widgets["hover_info_container"] = hover_container
        layout.addWidget(hover_container)
        self._renderer_control_widgets.append(hover_container)

        # === CUTAWAY (pygfx only): axis-aligned clipping planes ===
        # Three checkbox + slider + spin + flip-button rows, one per axis.
        # Each slider drives a world-space plane position; flip reverses the
        # half-space that is kept. The position range is refreshed from the
        # live scene bounds whenever the panel syncs from the visualizer.
        cutaway_group = self.create_subgroup_box("Cutaway")
        cutaway_layout = QVBoxLayout(cutaway_group)
        cutaway_layout.setSpacing(4)

        for axis in ("x", "y", "z"):
            row = QHBoxLayout()
            cb = QCheckBox(f"{axis.upper()}")
            cb.setChecked(False)
            cb.setToolTip(
                f"Enable a clipping plane perpendicular to the {axis.upper()} "
                "axis. Drag the slider to move the cut, press Flip to keep "
                "the opposite side."
            )
            cb.toggled.connect(lambda checked, a=axis: self._on_clip_enabled_toggled(a, checked))
            self.widgets[f"clip_{axis}_cb"] = cb
            row.addWidget(cb)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(-1000, 1000)
            slider.setValue(0)
            slider.setToolTip(f"Cutaway plane position along {axis.upper()} (meters)")
            slider.valueChanged.connect(
                lambda value, a=axis: self._on_clip_slider_changed(a, value)
            )
            self.widgets[f"clip_{axis}_slider"] = slider
            row.addWidget(slider)

            spin = QDoubleSpinBox()
            spin.setRange(-10000.0, 10000.0)
            spin.setDecimals(1)
            spin.setValue(0.0)
            spin.setSuffix(" m")
            spin.valueChanged.connect(lambda value, a=axis: self._on_clip_spin_changed(a, value))
            self.widgets[f"clip_{axis}_spin"] = spin
            row.addWidget(spin)

            flip_btn = QPushButton("Flip")
            flip_btn.setCheckable(True)
            flip_btn.setChecked(False)
            flip_btn.setFixedWidth(40)
            flip_btn.setToolTip("Keep the opposite half-space")
            flip_btn.toggled.connect(lambda checked, a=axis: self._on_clip_flip_toggled(a, checked))
            self.widgets[f"clip_{axis}_flip"] = flip_btn
            row.addWidget(flip_btn)

            cutaway_layout.addLayout(row)

        reset_row = QHBoxLayout()
        reset_btn = QPushButton("Reset Cutaway")
        reset_btn.setToolTip("Disable all clipping planes")
        reset_btn.clicked.connect(self._on_clip_reset_clicked)
        self.widgets["clip_reset_btn"] = reset_btn
        reset_row.addWidget(reset_btn)
        reset_row.addStretch()
        cutaway_layout.addLayout(reset_row)
        cutaway_layout.addWidget(create_contour_controls(self))

        layout.addWidget(cutaway_group)
        self._clipping_widgets.append(cutaway_group)

        return group

    def create_image_quality_panel(self) -> QWidget:
        """Create live rendering quality controls, independent of file export."""
        group = self.create_group_box("Image quality")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 4, 0, 4)

        aa_row = QHBoxLayout()
        aa_row.addWidget(QLabel("Antialiasing:"))
        self.widgets["aa_combo"] = QComboBox()
        for label in ("Off", "FXAA"):
            self.widgets["aa_combo"].addItem(label)
        self.widgets["aa_combo"].setToolTip(
            "FXAA smooths edges in the live view with a small amount of softness. "
            "Off preserves unfiltered pixels."
        )
        self.widgets["aa_combo"].currentTextChanged.connect(self._on_aa_mode_changed)
        aa_row.addWidget(self.widgets["aa_combo"])
        aa_row.addStretch()
        aa_container = self._make_row_container(aa_row)
        self.widgets["aa_container"] = aa_container
        layout.addWidget(aa_container)
        self.widgets["depth_precision_cb"] = QCheckBox("Improve distant depth precision")
        self.widgets["depth_precision_cb"].setToolTip(
            "Reduce distant water/ground and nearby-surface flicker by adjusting "
            "camera depth precision as you move. Adds a scene-bounds check; "
            "does not repair intersecting geometry or change ray tracing. "
            "Explicit camera clipping ranges are respected."
        )
        self.widgets["depth_precision_cb"].toggled.connect(self._on_depth_precision_toggled)
        layout.addWidget(self.widgets["depth_precision_cb"])

        self.widgets["culling_cb"] = QCheckBox("Cull geometry outside the camera view")
        self.widgets["culling_cb"].setChecked(False)
        self.widgets["culling_cb"].setToolTip(
            "Enable view frustum culling (hides geometry outside camera view)"
        )
        self.widgets["culling_cb"].toggled.connect(self._on_culling_toggled)
        layout.addWidget(self.widgets["culling_cb"])
        self._quality_widgets.append(group)
        return group

    def create_viewport_hud_panel(self) -> QWidget:
        """Create pygfx viewport overlay visibility and detail controls."""
        group = self.create_group_box("Viewport HUD")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Style:"))
        style_combo = QComboBox()
        for label, style in (("Classic", "classic"), ("Card", "card"), ("Minimal", "minimal")):
            style_combo.addItem(label, style)
        style_combo.setToolTip(
            "Classic uses translucent dark panels. Card uses the light surface and navy text "
            "of actor callout cards. Minimal uses compact ink panels. "
            "The style changes appearance without changing visible content."
        )
        style_combo.currentIndexChanged.connect(self._on_viewport_hud_style_changed)
        self.widgets["viewport_hud_style_combo"] = style_combo
        style_row.addWidget(style_combo)
        style_row.addStretch()
        layout.addLayout(style_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Legend/filter detail:"))
        mode_combo = QComboBox()
        for label, mode in (
            ("Compact", "compact"),
            ("Detailed", "detailed"),
        ):
            mode_combo.addItem(label, mode)
        mode_combo.setToolTip(
            "Compact summarizes active filters and shows up to 4 material entries. "
            "Detailed shows filter limits and up to 12 material entries. "
            "Other legends and status cues are the same in both modes. "
            "Use the persistent HUD control or Ctrl+H to show or hide all overlays."
        )
        mode_combo.currentIndexChanged.connect(self._on_viewport_hud_mode_changed)
        self.widgets["viewport_hud_mode_combo"] = mode_combo
        mode_row.addWidget(mode_combo)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        categories = (
            ("status", "Status", "Known simulation time, cutaway and historical-path cues"),
            ("legends", "Legends", "Coverage, MPC, RF X-Ray, trajectory, and marker legends"),
            (
                "filters",
                "Filters",
                "Active path-filter summary, including selected-material color swatches",
            ),
            ("annotations", "Hover", "Transient hover/tool-tip annotations"),
        )
        category_row = QHBoxLayout()
        for key, label, tooltip in categories:
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            checkbox.setToolTip(tooltip)
            checkbox.toggled.connect(
                lambda checked, name=key: self._on_viewport_hud_category_changed(name, checked)
            )
            self.widgets[f"viewport_hud_{key}_cb"] = checkbox
            category_row.addWidget(checkbox)
        category_row.addStretch()
        layout.addLayout(category_row)

        hint = QLabel("Ctrl+H toggles every viewport HUD overlay without changing scene layers.")
        hint.setProperty("role", "muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._viewport_hud_widgets.append(group)
        self._sync_viewport_hud_controls()
        return group

    # Sub-panel: Lighting

    def create_lighting_panel(self) -> QWidget:
        """Create the lighting sub-panel (shader, skybox, IBL, light rig)."""
        group = self.create_group_box("Lighting and environment")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(create_dusk_controls(self))

        shader_group = self.create_subgroup_box("Shader")
        shader_layout = QHBoxLayout(shader_group)
        shader_layout.setSpacing(4)
        shader_layout.setContentsMargins(6, 6, 6, 6)
        shader_row = QHBoxLayout()
        shader_row.addWidget(QLabel("Shader:"))
        self.widgets["shader_combo"] = QComboBox()
        for shader in ["Standard", "Unlit", "Normals"]:
            self.widgets["shader_combo"].addItem(shader)
        self.widgets["shader_combo"].setToolTip(
            "Standard: Full PBR rendering with lighting\n"
            "Unlit: Colors only, no lighting effects\n"
            "Normals: Visualize surface normal directions"
        )
        self.widgets["shader_combo"].currentTextChanged.connect(self._on_shader_changed)
        shader_row.addWidget(self.widgets["shader_combo"])
        shader_row.addStretch()
        shader_layout.addLayout(shader_row)
        layout.addWidget(shader_group)
        self.widgets["shader_group"] = shader_group
        self._open3d_scene_shader_widgets.append(shader_group)

        background_row = QHBoxLayout()
        background_row.addWidget(QLabel("Background:"))
        self.widgets["background_solid_radio"] = QRadioButton("Solid color")
        # Keep the established key used by appearance snapshots and Paper mode.
        self.widgets["skybox_cb"] = QRadioButton("Skybox")
        modes = QButtonGroup(group)
        modes.addButton(self.widgets["background_solid_radio"])
        modes.addButton(self.widgets["skybox_cb"])
        self._background_modes = modes
        self.widgets["background_solid_radio"].setChecked(True)
        self.widgets["background_solid_radio"].setToolTip(
            "Use a solid backdrop. Environment lighting and reflections remain active."
        )
        self.widgets["skybox_cb"].setToolTip(
            "Show the selected environment image behind the scene."
        )
        self.widgets["skybox_cb"].toggled.connect(self._on_skybox_toggled)
        background_row.addWidget(self.widgets["background_solid_radio"])
        background_row.addWidget(self.widgets["skybox_cb"])
        background_row.addStretch()
        layout.addLayout(background_row)

        environment_group = CollapsibleSection(
            "Environment lighting (IBL)", start_open=False, flat_header=True
        )
        environment_group.setToolTip(
            "The environment provides lighting and reflections with either background. "
            "Selecting Skybox also displays its image."
        )
        environment_layout = environment_group.content_layout()
        environment_layout.setSpacing(4)
        environment_layout.setContentsMargins(6, 6, 6, 6)

        # Background preset (both renderers)
        bg_row = QHBoxLayout()
        bg_row.addWidget(QLabel("Color:"))
        self.widgets["bg_combo"] = QComboBox()
        for name in self._BG_PRESETS:
            self.widgets["bg_combo"].addItem(name)
        self.widgets["bg_combo"].setToolTip("Scene background color preset")
        self.widgets["bg_combo"].currentTextChanged.connect(self._on_background_changed)
        bg_row.addWidget(self.widgets["bg_combo"])
        bg_row.addStretch()
        self.widgets["background_color_row"] = self._make_row_container(bg_row)
        layout.addWidget(self.widgets["background_color_row"])

        ibl_row = QHBoxLayout()
        ibl_row.addWidget(QLabel("Map:"))
        self.widgets["ibl_combo"] = QComboBox()
        with QSignalBlocker(self.widgets["ibl_combo"]):
            for name in self._ibl_library:
                self.widgets["ibl_combo"].addItem(name)
        self.widgets["ibl_combo"].setToolTip("Select an HDRI/IBL environment for lighting")
        self.widgets["ibl_combo"].currentTextChanged.connect(self._on_ibl_changed)
        ibl_row.addWidget(self.widgets["ibl_combo"], 1)
        self.widgets["ibl_load_btn"] = QPushButton("Load HDRI...")
        self.widgets["ibl_load_btn"].setToolTip("Load a custom HDR/EXR environment")
        self.widgets["ibl_load_btn"].clicked.connect(self._on_load_ibl)
        # Load stays beside the selector without forcing the sidebar wider.
        self.widgets["ibl_load_btn"].setText("Load…")
        ibl_row.addWidget(self.widgets["ibl_load_btn"])
        environment_layout.addLayout(ibl_row)

        self.widgets["environment_download"] = EnvironmentDownload(environment_group)
        self.widgets["environment_download"].ready.connect(self._apply_optional_environment)
        environment_layout.addWidget(self.widgets["environment_download"])

        ibl_ctrl_row = QHBoxLayout()
        ibl_ctrl_row.addWidget(QLabel("Intensity:"))
        self.widgets["ibl_intensity_slider"] = QSlider(Qt.Horizontal)
        self.widgets["ibl_intensity_slider"].setRange(0, 100)
        self.widgets["ibl_intensity_slider"].setValue(30)
        self.widgets["ibl_intensity_slider"].setToolTip(
            "Environment map brightness for IBL reflections"
        )
        self.widgets["ibl_intensity_slider"].valueChanged.connect(self._on_ibl_slider_changed)
        ibl_ctrl_row.addWidget(self.widgets["ibl_intensity_slider"])
        self.widgets["ibl_intensity_spin"] = QSpinBox()
        self.widgets["ibl_intensity_spin"].setRange(0, 100000)
        self.widgets["ibl_intensity_spin"].setSingleStep(5000)
        self.widgets["ibl_intensity_spin"].setValue(30000)
        self.widgets["ibl_intensity_spin"].valueChanged.connect(self._on_ibl_spin_changed)
        ibl_ctrl_row.addWidget(self.widgets["ibl_intensity_spin"])
        environment_layout.addLayout(ibl_ctrl_row)
        layout.addWidget(environment_group)
        self.widgets["environment_group"] = environment_group
        self._environment_light_widgets.append(environment_group)

        direct_group = CollapsibleSection("Direct lights", start_open=False, flat_header=True)
        direct_layout = direct_group.content_layout()
        direct_layout.setSpacing(4)
        direct_layout.setContentsMargins(6, 6, 6, 6)

        profile_row = QHBoxLayout()
        self.widgets["lighting_profile_label"] = QLabel("Lighting preset:")
        profile_row.addWidget(self.widgets["lighting_profile_label"])
        self.widgets["lighting_profile_combo"] = QComboBox()
        self.widgets["lighting_profile_combo"].setToolTip(
            "Set ambient, key, fill and camera-follow lights, their directions, "
            "environment intensity and shadow settings together. "
            "The environment image and background choice stay unchanged."
        )
        self.widgets["lighting_profile_combo"].currentIndexChanged.connect(
            self._on_lighting_profile_changed
        )
        profile_row.addWidget(self.widgets["lighting_profile_combo"], 1)
        layout.addLayout(profile_row)

        head_row = QHBoxLayout()
        self.widgets["headlight_cb"] = QCheckBox("Headlight")
        self.widgets["headlight_cb"].setChecked(True)
        self.widgets["headlight_cb"].setToolTip("Camera-follow light for stable highlights")
        self.widgets["headlight_cb"].toggled.connect(self._on_headlight_toggled)
        head_row.addWidget(self.widgets["headlight_cb"])
        head_row.addWidget(QLabel("Intensity:"))
        self.widgets["headlight_intensity_spin"] = QDoubleSpinBox()
        self.widgets["headlight_intensity_spin"].setDecimals(2)
        self.widgets["headlight_intensity_spin"].setRange(0.0, 20.0)
        self.widgets["headlight_intensity_spin"].setSingleStep(0.1)
        self.widgets["headlight_intensity_spin"].setValue(1.2)
        self.widgets["headlight_intensity_spin"].valueChanged.connect(
            self._on_headlight_intensity_changed
        )
        head_row.addWidget(self.widgets["headlight_intensity_spin"])
        head_row.addStretch()
        direct_layout.addLayout(head_row)

        self.widgets["key_light_az_spin"] = QDoubleSpinBox()
        self.widgets["key_light_az_spin"].setAccessibleName("Key light azimuth")
        self.widgets["key_light_az_spin"].setToolTip("Key light azimuth" + " in degrees")
        self.widgets["key_light_az_spin"].setDecimals(1)
        self.widgets["key_light_az_spin"].setRange(-180.0, 180.0)
        self.widgets["key_light_az_spin"].setSingleStep(5.0)
        self.widgets["key_light_az_spin"].setValue(-123.0)
        self.widgets["key_light_az_spin"].valueChanged.connect(self._on_key_light_angles_changed)
        self.widgets["key_light_el_spin"] = QDoubleSpinBox()
        self.widgets["key_light_el_spin"].setAccessibleName("Key light elevation")
        self.widgets["key_light_el_spin"].setToolTip("Key light elevation" + " in degrees")
        self.widgets["key_light_el_spin"].setDecimals(1)
        self.widgets["key_light_el_spin"].setRange(-89.0, 89.0)
        self.widgets["key_light_el_spin"].setSingleStep(2.0)
        self.widgets["key_light_el_spin"].setValue(-48.0)
        self.widgets["key_light_el_spin"].valueChanged.connect(self._on_key_light_angles_changed)
        self.widgets["key_light_intensity_spin"] = QDoubleSpinBox()
        self.widgets["key_light_intensity_spin"].setAccessibleName("Key light intensity")
        self.widgets["key_light_intensity_spin"].setToolTip("Key light intensity")
        self.widgets["key_light_intensity_spin"].setDecimals(2)
        self.widgets["key_light_intensity_spin"].setRange(0.0, 50.0)
        self.widgets["key_light_intensity_spin"].setSingleStep(0.1)
        self.widgets["key_light_intensity_spin"].setValue(3.0)
        self.widgets["key_light_intensity_spin"].valueChanged.connect(
            self._on_key_light_intensity_changed
        )

        self.widgets["fill_light_az_spin"] = QDoubleSpinBox()
        self.widgets["fill_light_az_spin"].setAccessibleName("Fill light azimuth")
        self.widgets["fill_light_az_spin"].setToolTip("Fill light azimuth" + " in degrees")
        self.widgets["fill_light_az_spin"].setDecimals(1)
        self.widgets["fill_light_az_spin"].setRange(-180.0, 180.0)
        self.widgets["fill_light_az_spin"].setSingleStep(5.0)
        self.widgets["fill_light_az_spin"].setValue(53.0)
        self.widgets["fill_light_az_spin"].valueChanged.connect(self._on_fill_light_angles_changed)
        self.widgets["fill_light_el_spin"] = QDoubleSpinBox()
        self.widgets["fill_light_el_spin"].setAccessibleName("Fill light elevation")
        self.widgets["fill_light_el_spin"].setToolTip("Fill light elevation" + " in degrees")
        self.widgets["fill_light_el_spin"].setDecimals(1)
        self.widgets["fill_light_el_spin"].setRange(-89.0, 89.0)
        self.widgets["fill_light_el_spin"].setSingleStep(2.0)
        self.widgets["fill_light_el_spin"].setValue(-45.0)
        self.widgets["fill_light_el_spin"].valueChanged.connect(self._on_fill_light_angles_changed)
        self.widgets["fill_light_intensity_spin"] = QDoubleSpinBox()
        self.widgets["fill_light_intensity_spin"].setAccessibleName("Fill light intensity")
        self.widgets["fill_light_intensity_spin"].setToolTip("Fill light intensity")
        self.widgets["fill_light_intensity_spin"].setDecimals(2)
        self.widgets["fill_light_intensity_spin"].setRange(0.0, 50.0)
        self.widgets["fill_light_intensity_spin"].setSingleStep(0.1)
        self.widgets["fill_light_intensity_spin"].setValue(1.0)
        self.widgets["fill_light_intensity_spin"].valueChanged.connect(
            self._on_fill_light_intensity_changed
        )
        light_grid = QGridLayout()
        light_grid.setHorizontalSpacing(8)
        for column, title in enumerate(("Light", "Azimuth °", "Elevation °", "Intensity")):
            light_grid.addWidget(QLabel(title), 0, column)
        for row, prefix in enumerate(("key", "fill"), start=1):
            light_grid.addWidget(QLabel(prefix.title()), row, 0)
            for column, field in enumerate(("az", "el", "intensity"), start=1):
                light_grid.addWidget(self.widgets[f"{prefix}_light_{field}_spin"], row, column)
        direct_layout.addLayout(light_grid)
        layout.addWidget(direct_group)
        self.widgets["direct_lights_group"] = direct_group
        self._pygfx_light_widgets.append(direct_group)

        advanced_group = self.create_subgroup_box("Shadows")
        advanced_layout = QVBoxLayout(advanced_group)
        advanced_layout.setSpacing(4)
        advanced_layout.setContentsMargins(6, 6, 6, 6)
        self.widgets["shadows_cb"] = QCheckBox("Cast shadows")
        self.widgets["shadows_cb"].setChecked(True)
        self.widgets["shadows_cb"].setToolTip("Enable/disable shadow mapping")
        self.widgets["shadows_cb"].toggled.connect(self._on_shadows_toggled)
        advanced_layout.addWidget(self.widgets["shadows_cb"])

        self.widgets["o3d_settings_cb"] = QCheckBox("Show O3D Settings")
        self.widgets["o3d_settings_cb"].setChecked(False)
        self.widgets["o3d_settings_cb"].setToolTip(
            "Show Open3D's built-in settings panel for advanced\n"
            "lighting controls (sun color, sun follows camera, etc.)"
        )
        self.widgets["o3d_settings_cb"].toggled.connect(self._on_o3d_settings_toggled)
        advanced_layout.addWidget(self.widgets["o3d_settings_cb"])
        layout.addWidget(advanced_group)
        self.widgets["advanced_lighting_group"] = advanced_group
        self._advanced_light_widgets.append(advanced_group)

        shadow_group = CollapsibleSection("Shadow detail", start_open=False, flat_header=True)
        shadow_layout = shadow_group.content_layout()
        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("Coverage:"))
        quality = QComboBox()
        quality.addItem("Standard (whole scene)", "standard")
        quality.addItem("Fitted scene", "scene")
        quality.addItem("Inspection focus", "focus")
        quality.setToolTip(
            "Fitted scene uses the native 1024² map more efficiently. "
            "Inspection focus limits shadow coverage to a pinned region; "
            "use Refocus here after moving to a different area."
        )
        quality.currentIndexChanged.connect(self._on_shadow_quality_changed)
        quality_row.addWidget(quality, 1)
        self.widgets["shadow_quality_combo"] = quality
        shadow_layout.addLayout(quality_row)

        focus_row = QHBoxLayout()
        focus_label = QLabel("Focus radius:")
        focus_row.addWidget(focus_label)
        self.widgets["shadow_focus_label"] = focus_label
        radius = QDoubleSpinBox()
        radius.setRange(0.1, 100000.0)
        radius.setDecimals(1)
        radius.setValue(25.0)
        radius.setSuffix(" m")
        radius.setToolTip("Half-size of a pinned world-space cube around the look-at point")
        radius.valueChanged.connect(self._on_shadow_focus_radius_changed)
        self.widgets["shadow_focus_radius_spin"] = radius
        focus_row.addWidget(radius)
        refocus = QPushButton("Refocus here")
        refocus.setToolTip("Move the shadow focus to the current camera look-at point")
        refocus.clicked.connect(self._on_shadow_refocus)
        self.widgets["shadow_refocus_btn"] = refocus
        focus_row.addWidget(refocus)
        shadow_layout.addLayout(focus_row)

        bias_row = QHBoxLayout()
        bias_row.addWidget(QLabel("Depth offset:"))
        bias = QDoubleSpinBox()
        bias.setRange(0.0, 8.0)
        bias.setDecimals(2)
        bias.setSingleStep(0.25)
        bias.setValue(1.5)
        bias.setSuffix(" texels")
        bias.setToolTip(
            "Fitted modes only. Increase to reduce self-shadow stripes; "
            "decrease if shadows appear detached from objects."
        )
        bias.valueChanged.connect(self._on_shadow_bias_changed)
        self.widgets["shadow_bias_spin"] = bias
        bias_row.addWidget(bias)
        shadow_layout.addLayout(bias_row)
        summary = QLabel("Native shadow map: 1024 × 1024")
        summary.setWordWrap(True)
        self.widgets["shadow_detail_summary"] = summary
        shadow_layout.addWidget(summary)
        self.widgets["shadow_detail_group"] = shadow_group
        # Metadata only: the canvas still exclusively owns frame rendering.
        summary_timer = QTimer(shadow_group)
        summary_timer.setInterval(1000)
        summary_timer.timeout.connect(self._refresh_shadow_summary)
        summary_timer.start()
        self._shadow_summary_timer = summary_timer
        advanced_layout.addWidget(shadow_group)

        return group

    # Inline controls: Edge width

    def _create_edge_width_controls(self) -> QWidget:
        """Create compact scene-edge width controls for the Edges row."""
        # Edge line width control (pygfx only; Open3D exposes line width globally).
        edge_line_row = QHBoxLayout()
        edge_line_row.setSpacing(4)
        self._edge_line_label = QLabel("Width:")
        edge_line_row.addWidget(self._edge_line_label)
        self.widgets["edge_line_width_slider"] = QSlider(Qt.Horizontal)
        self.widgets["edge_line_width_slider"].setRange(1, 10)
        self.widgets["edge_line_width_slider"].setValue(1)
        self.widgets["edge_line_width_slider"].setToolTip(
            "Width of scene wireframe edge lines in pixels"
        )
        self.widgets["edge_line_width_slider"].valueChanged.connect(
            self._on_edge_line_width_slider_changed
        )
        edge_line_row.addWidget(self.widgets["edge_line_width_slider"])
        self.widgets["edge_line_width_spin"] = QSpinBox()
        self.widgets["edge_line_width_spin"].setRange(1, 10)
        self.widgets["edge_line_width_spin"].setValue(1)
        self.widgets["edge_line_width_spin"].setFixedWidth(50)
        self.widgets["edge_line_width_spin"].valueChanged.connect(
            self._on_edge_line_width_spin_changed
        )
        edge_line_row.addWidget(self.widgets["edge_line_width_spin"])
        edge_line_container = self._make_row_container(edge_line_row)
        self.widgets["edge_line_width_container"] = edge_line_container
        self._pygfx_renderer_widgets.append(edge_line_container)

        return edge_line_container

    # Sync helpers

    def sync_presentation_effects(self) -> None:
        """Refresh optional effect availability after a scenario is configured."""
        sync_effect_controls(self)

    def sync_scene_guides(self) -> None:
        """Refresh guide availability after geographic metadata changes."""
        sync_guide_controls(self)

    def _sync_from_visualizer(self) -> None:
        """Sync widget values and visibility from visualizer and renderer state."""
        viz = self.parent
        if viz is None:
            return

        renderer = getattr(viz, "renderer", None)
        app_state = getattr(viz, "app_state", None)
        capabilities = renderer_capabilities(renderer)

        sync_effect_controls(self)
        sync_guide_controls(self)

        # Capability flags decide visibility; individual callbacks still guard
        # protocol hooks because tests and startup can provide partial renderers.
        renderer_available = renderer is not None
        supports_wireframe = renderer is not None and capabilities.wireframe
        supports_axes = renderer is not None and capabilities.axes
        supports_skybox = renderer is not None and capabilities.skybox
        supports_screen_space_labels = renderer is not None and capabilities.screen_space_labels
        supports_scene_shader = renderer is not None and capabilities.scene_shader
        supports_frustum_culling = renderer is not None and capabilities.frustum_culling
        supports_shadow_toggle = renderer is not None and capabilities.shadow_toggle
        supports_o3d_settings_panel = renderer is not None and capabilities.open3d_settings_panel
        supports_ibl = renderer is not None and capabilities.ibl
        supports_clipping_planes = renderer is not None and capabilities.clipping_planes
        supports_hover_info = renderer is not None and capabilities.hover_info
        supports_antialiasing = renderer is not None and capabilities.antialiasing
        supports_ground_grid = renderer is not None and capabilities.ground_grid
        supports_viewport_hud = renderer is not None and capabilities.viewport_hud

        wireframe_cb = self.widgets.get("wireframe_cb")
        if wireframe_cb is not None and supports_wireframe and hasattr(renderer, "get_wireframe"):
            with QSignalBlocker(wireframe_cb):
                wireframe_cb.setChecked(renderer.get_wireframe())

        # Sync outline checkboxes
        for key, attr in (
            ("building_alpha_slider", "current_building_alpha"),
            ("target_alpha_slider", "current_target_alpha"),
        ):
            slider = self.widgets.get(key)
            if slider is not None:
                with QSignalBlocker(slider):
                    slider.setValue(round(float(getattr(viz, attr, 1.0)) * 100))
        outline_cb = self.widgets.get("outline_cb")
        if outline_cb is not None and hasattr(viz, "outlines_enabled"):
            with QSignalBlocker(outline_cb):
                outline_cb.setChecked(bool(viz.outlines_enabled))

        target_outline_cb = self.widgets.get("target_outline_cb")
        if target_outline_cb is not None and hasattr(viz, "target_outlines_enabled"):
            with QSignalBlocker(target_outline_cb):
                target_outline_cb.setChecked(bool(viz.target_outlines_enabled))

        # Sync background combo
        bg_combo = self.widgets.get("bg_combo")
        if bg_combo is not None and hasattr(viz, "current_background_preset"):
            current = viz.current_background_preset or DEFAULT_SCENE_BACKGROUND_PRESET
            with QSignalBlocker(bg_combo):
                if current in self._BG_PRESETS:
                    bg_combo.setCurrentText(current)
                else:
                    color = getattr(viz, "current_background_color", None)
                    if color is not None:
                        for name, preset in self._BG_PRESETS.items():
                            if np.allclose(preset, color, atol=1e-3):
                                bg_combo.setCurrentText(name)
                                break

        # Sync IBL widgets from the renderer state. Open3D and pygfx have
        # different defaults, so the panel should reflect the active renderer
        # instead of keeping the generic widget seed value.
        if supports_ibl and hasattr(renderer, "get_ibl_intensity"):
            intensity = renderer.get_ibl_intensity()
            if intensity is not None:
                self._sync_ibl_widgets(int(intensity))

        depth_cb = self.widgets.get("depth_precision_cb")
        if depth_cb is not None:
            supported = renderer is not None and capabilities.depth_precision
            depth_cb.setVisible(supported)
            depth_cb.setEnabled(supported)
            if supported:
                with QSignalBlocker(depth_cb):
                    depth_cb.setChecked(renderer.get_depth_precision_enabled())

        mipmaps_cb = self.widgets.get("texture_mipmaps_cb")
        if mipmaps_cb is not None:
            supported = renderer is not None and capabilities.texture_mipmaps
            mipmaps_cb.setVisible(supported)
            from ..materials.texture_policy import textures_globally_enabled

            mipmaps_cb.setEnabled(supported and textures_globally_enabled())
            if supported:
                with QSignalBlocker(mipmaps_cb):
                    mipmaps_cb.setChecked(renderer.get_texture_mipmaps_enabled())

        if app_state is not None:
            screen_space_cb = self.widgets.get("screen_space_labels_cb")
            if screen_space_cb is not None:
                with QSignalBlocker(screen_space_cb):
                    screen_space_cb.setChecked(bool(getattr(app_state, "label_screen_space", True)))

            aa_combo = self.widgets.get("aa_combo")
            if aa_combo is not None:
                aa_mode = (
                    renderer.get_antialiasing_mode()
                    if supports_antialiasing and hasattr(renderer, "get_antialiasing_mode")
                    else str(getattr(app_state, "antialiasing_mode", "off"))
                )
                aa_label = {
                    "off": "Off",
                    "fxaa": "FXAA",
                    "ddaa": "FXAA",
                    "ppaa": "FXAA",
                }.get(str(aa_mode).lower(), "Off")
                with QSignalBlocker(aa_combo):
                    aa_combo.setCurrentText(aa_label)

            grid_cb = self.widgets.get("ground_grid_cb")
            if grid_cb is not None:
                grid_enabled = (
                    renderer.get_ground_grid_visible()
                    if supports_ground_grid and hasattr(renderer, "get_ground_grid_visible")
                    else bool(getattr(app_state, "show_ground_grid", False))
                )
                with QSignalBlocker(grid_cb):
                    grid_cb.setChecked(bool(grid_enabled))

            hover_combo = self.widgets.get("hover_info_combo")
            if hover_combo is not None:
                hover_mode = (
                    renderer.get_hover_info_mode()
                    if supports_hover_info and hasattr(renderer, "get_hover_info_mode")
                    else str(getattr(app_state, "hover_info_mode", "essential"))
                )
                hover_label = self._HOVER_INFO_LABEL_BY_MODE.get(
                    str(hover_mode).lower(), "Essential"
                )
                with QSignalBlocker(hover_combo):
                    hover_combo.setCurrentText(hover_label)

            self._sync_viewport_hud_controls()

            for axis in ("x", "y", "z"):
                enabled = bool(getattr(app_state, f"clip_{axis}_enabled", False))
                position = float(getattr(app_state, f"clip_{axis}_position", 0.0))
                flipped = bool(getattr(app_state, f"clip_{axis}_flip", False))

                cb = self.widgets.get(f"clip_{axis}_cb")
                if cb is not None:
                    with QSignalBlocker(cb):
                        cb.setChecked(enabled)

                slider = self.widgets.get(f"clip_{axis}_slider")
                if slider is not None:
                    clamped = int(max(slider.minimum(), min(slider.maximum(), round(position))))
                    with QSignalBlocker(slider):
                        slider.setValue(clamped)

                spin = self.widgets.get(f"clip_{axis}_spin")
                if spin is not None:
                    with QSignalBlocker(spin):
                        spin.setValue(position)

                flip = self.widgets.get(f"clip_{axis}_flip")
                if flip is not None:
                    with QSignalBlocker(flip):
                        flip.setChecked(flipped)

        if renderer is not None and supports_ibl:
            self._refresh_ibl_library(renderer)
            ibl_name = renderer.get_ibl_name() if hasattr(renderer, "get_ibl_name") else None
            if ibl_name:
                self._sync_ibl_selection(ibl_name)
        if renderer is not None and capabilities.direct_lighting:
            self._sync_light_rig_widgets(renderer)

        # Update widget visibility/enabled state based on renderer
        self._update_renderer_visibility(
            renderer_available=renderer_available,
            supports_scene_shader=supports_scene_shader,
            supports_frustum_culling=supports_frustum_culling,
            supports_shadow_toggle=supports_shadow_toggle,
            supports_o3d_settings_panel=supports_o3d_settings_panel,
            supports_ibl=supports_ibl,
            supports_wireframe=supports_wireframe,
            supports_skybox=supports_skybox,
            supports_axes=supports_axes,
            supports_screen_space_labels=supports_screen_space_labels,
            supports_clipping_planes=supports_clipping_planes,
            supports_hover_info=supports_hover_info,
            supports_antialiasing=supports_antialiasing,
            supports_ground_grid=supports_ground_grid,
            supports_viewport_hud=supports_viewport_hud,
        )
        self._update_light_rig_visibility(renderer)

        # Fit cutaway slider ranges from live scene bounds. Pygfx only
        # Open3D keeps the Cutaway group hidden via supports_clipping_planes.
        if renderer is not None and capabilities.clipping_planes:
            self._sync_cutaway_ranges(renderer)

        # Backdrop radios and disclosure follow the actual renderer, including
        # appearance presets and Demo restoration that bypass widget signals.
        if supports_skybox:
            skybox_cb = self.widgets.get("skybox_cb")
            if skybox_cb is not None and callable(getattr(renderer, "get_skybox_visible", None)):
                self._sync_background_controls(renderer.get_skybox_visible())
            elif skybox_cb is not None and skybox_cb.isChecked():
                viz.renderer.show_skybox(True)
                self._sync_background_controls()
        else:
            self._sync_background_controls(False)

    def restore_session_state(self, state: Dict[str, Any]) -> None:
        """Apply saved renderer controls once and mirror widgets without signals."""
        if not isinstance(state, dict):
            return

        def _integer(key: str) -> Optional[int]:
            if key not in state:
                return None
            try:
                return int(state[key])
            except (TypeError, ValueError):
                return None

        common: Dict[str, Any] = {}
        background = state.get("background_preset")
        if background in self._BG_PRESETS:
            common["bg_combo"] = str(background)
        if "show_edges" in state:
            common["outline_cb"] = bool(state["show_edges"])
        if common:
            self._apply_settings_to_widgets(common)

        viz = self.parent
        renderer = getattr(viz, "renderer", None) if viz is not None else None

        def _set_pair(slider_key: str, spin_key: str, value: Optional[int]) -> None:
            if value is None:
                return
            widgets = [
                widget
                for widget in (self.widgets.get(slider_key), self.widgets.get(spin_key))
                if widget is not None
            ]
            blockers = [QSignalBlocker(widget) for widget in widgets]
            try:
                for widget in widgets:
                    widget.setValue(value)
            finally:
                blockers.clear()

        edge_width = _integer("edge_line_width")
        _set_pair("edge_line_width_slider", "edge_line_width_spin", edge_width)
        if (
            edge_width is not None
            and renderer is not None
            and renderer_capabilities(renderer).wireframe
            and hasattr(renderer, "set_edge_line_width")
        ):
            renderer.set_edge_line_width(float(edge_width))

        for key, widget_key in (
            ("building_alpha", "building_alpha_slider"),
            ("target_alpha", "target_alpha_slider"),
        ):
            if key not in state:
                continue
            try:
                value = int(round(float(state[key]) * 100.0))
            except (TypeError, ValueError):
                continue
            widget = self.widgets.get(widget_key)
            if widget is not None:
                with QSignalBlocker(widget):
                    widget.setValue(value)

        if "show_target_edges" in state:
            enabled = bool(state["show_target_edges"])
            widget = self.widgets.get("target_outline_cb")
            if widget is not None:
                with QSignalBlocker(widget):
                    widget.setChecked(enabled)
            if viz is not None:
                viz.target_service.set_target_edge_visibility(enabled)

        if "show_axes" in state:
            enabled = bool(state["show_axes"])
            widget = self.widgets.get("show_axes_cb")
            if widget is not None:
                with QSignalBlocker(widget):
                    widget.setChecked(enabled)
            if renderer is not None and renderer_capabilities(renderer).axes:
                renderer.show_axes(enabled)
        restore_guide_controls(self, state)

    def _update_renderer_visibility(
        self,
        *,
        renderer_available: bool,
        supports_scene_shader: bool,
        supports_frustum_culling: bool,
        supports_shadow_toggle: bool,
        supports_o3d_settings_panel: bool,
        supports_ibl: bool,
        supports_skybox: bool,
        supports_wireframe: bool,
        supports_axes: bool,
        supports_screen_space_labels: bool,
        supports_clipping_planes: bool,
        supports_hover_info: bool,
        supports_antialiasing: bool,
        supports_ground_grid: bool,
        supports_viewport_hud: bool,
    ) -> None:
        """Show/hide renderer-specific widgets based on active renderer."""
        for w in self._renderer_control_widgets:
            w.setVisible(renderer_available)
        for w in self._pygfx_renderer_widgets:
            w.setVisible(supports_wireframe)

        for w in self._clipping_widgets:
            w.setVisible(supports_clipping_planes)

        for w in self._quality_widgets:
            w.setVisible(renderer_available)
        for w in self._viewport_hud_widgets:
            w.setVisible(renderer_available and supports_viewport_hud)

        aa_container = self.widgets.get("aa_container")
        if aa_container is not None:
            aa_container.setVisible(supports_antialiasing)

        ground_grid_cb = self.widgets.get("ground_grid_cb")
        if ground_grid_cb is not None:
            ground_grid_cb.setVisible(supports_ground_grid)

        show_axes_cb = self.widgets.get("show_axes_cb")
        if show_axes_cb is not None:
            show_axes_cb.setVisible(renderer_available and supports_axes)

        screen_space_labels_cb = self.widgets.get("screen_space_labels_cb")
        if screen_space_labels_cb is not None:
            screen_space_labels_cb.setVisible(renderer_available and supports_screen_space_labels)

        hover_info_container = self.widgets.get("hover_info_container")
        if hover_info_container is not None:
            hover_info_container.setVisible(renderer_available and supports_hover_info)

        culling_cb = self.widgets.get("culling_cb")
        if culling_cb is not None:
            culling_cb.setVisible(renderer_available and supports_frustum_culling)

        shadows_cb = self.widgets.get("shadows_cb")
        if shadows_cb is not None:
            shadows_cb.setVisible(renderer_available and supports_shadow_toggle)

        o3d_cb = self.widgets.get("o3d_settings_cb")
        if o3d_cb is not None:
            o3d_cb.setVisible(renderer_available and supports_o3d_settings_panel)

        for w in self._open3d_scene_shader_widgets:
            w.setVisible(renderer_available and supports_scene_shader)

        environment_visible = renderer_available and (supports_ibl or supports_skybox)
        for w in self._environment_light_widgets:
            w.setVisible(environment_visible)

        for key in ("skybox_cb", "background_solid_radio"):
            button = self.widgets.get(key)
            if button is not None:
                button.setVisible(supports_skybox)

        for key in ("ibl_combo", "ibl_load_btn", "ibl_intensity_slider", "ibl_intensity_spin"):
            widget = self.widgets.get(key)
            if widget is not None:
                widget.setVisible(supports_ibl)

        advanced_visible = renderer_available and (
            supports_shadow_toggle or supports_o3d_settings_panel
        )
        for w in self._advanced_light_widgets:
            w.setVisible(advanced_visible)

    def _sync_cutaway_ranges(self, renderer: object) -> None:
        """Rescale cutaway sliders to the current scene bounds (pygfx only).

        Uses the renderer's live bounding box — so amass_showcase lands on
        ±20 m and dense city scenes on ±500 m without the user having to tweak
        anything.
        """
        compute = getattr(renderer, "compute_scene_bounds", None)
        if not callable(compute):
            return
        try:
            bounds = compute()
        except Exception:
            return
        if bounds is None:
            return
        try:
            mn = np.asarray(bounds.min_bound, dtype=np.float64)
            mx = np.asarray(bounds.max_bound, dtype=np.float64)
        except (AttributeError, TypeError, ValueError):
            return
        if mn.shape != (3,) or mx.shape != (3,):
            return

        for i, axis in enumerate(("x", "y", "z")):
            lo = int(np.floor(mn[i])) - 1
            hi = int(np.ceil(mx[i])) + 1
            if hi <= lo:
                continue
            slider = self.widgets.get(f"clip_{axis}_slider")
            spin = self.widgets.get(f"clip_{axis}_spin")
            if slider is not None:
                with QSignalBlocker(slider):
                    slider.setRange(lo, hi)
            if spin is not None:
                with QSignalBlocker(spin):
                    spin.setRange(float(lo) - 100.0, float(hi) + 100.0)

    def _update_light_rig_visibility(self, renderer: Optional[object]) -> None:
        """Show pygfx light controls only when the active renderer supports them."""
        capabilities = renderer_capabilities(renderer)
        has_light_rig = renderer is not None and capabilities.direct_lighting
        for widget in self._pygfx_light_widgets:
            widget.setVisible(has_light_rig)
        shadow_group = self.widgets.get("shadow_detail_group")
        if shadow_group is not None:
            shadow_group.setVisible(renderer is not None and capabilities.shadow_quality)
        has_profiles = has_light_rig and capabilities.lighting_profiles
        for key in ("lighting_profile_label", "lighting_profile_combo"):
            widget = self.widgets.get(key)
            if widget is not None:
                widget.setVisible(has_profiles)

    def _sync_light_rig_widgets(self, renderer: object) -> None:
        """Sync pygfx light rig widgets from renderer state when available."""
        if not renderer_capabilities(renderer).direct_lighting:
            return
        try:
            state = renderer.get_light_rig_state()
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return
        if not isinstance(state, dict):
            return

        def _set_cb(name: str, value: bool) -> None:
            """Set one checkbox from renderer state without echoing a signal."""
            w = self.widgets.get(name)
            if w is None:
                return
            with QSignalBlocker(w):
                w.setChecked(bool(value))

        def _set_spin(name: str, value: float) -> None:
            """Set one spinbox from renderer state without echoing a signal."""
            w = self.widgets.get(name)
            if w is None:
                return
            with QSignalBlocker(w):
                w.setValue(float(value))

        _set_cb("headlight_cb", bool(state.get("headlight_enabled", True)))
        _set_cb("shadows_cb", bool(state.get("shadows_enabled", True)))
        _set_spin("headlight_intensity_spin", float(state.get("headlight_intensity", 1.2)))
        _set_spin("key_light_az_spin", float(state.get("key_azimuth_deg", -123.0)))
        _set_spin("key_light_el_spin", float(state.get("key_elevation_deg", -48.0)))
        _set_spin("key_light_intensity_spin", float(state.get("key_intensity", 3.0)))
        _set_spin("fill_light_az_spin", float(state.get("fill_azimuth_deg", 53.0)))
        _set_spin("fill_light_el_spin", float(state.get("fill_elevation_deg", -45.0)))
        _set_spin("fill_light_intensity_spin", float(state.get("fill_intensity", 1.0)))
        self._sync_lighting_profile_widgets(renderer)
        self._sync_shadow_quality_widgets(renderer)

    def _sync_shadow_quality_widgets(self, renderer: object) -> None:
        """Mirror actual fitted coverage without triggering another fit."""
        if not renderer_capabilities(renderer).shadow_quality:
            return
        state = renderer.get_shadow_quality_state()
        combo = self.widgets.get("shadow_quality_combo")
        if combo is None:
            return
        with QSignalBlocker(combo):
            combo.setCurrentIndex(combo.findData(state["quality"]))
        for key, field in (
            ("shadow_focus_radius_spin", "focus_radius"),
            ("shadow_bias_spin", "bias_texels"),
        ):
            widget = self.widgets.get(key)
            if widget is not None:
                with QSignalBlocker(widget):
                    widget.setValue(float(state[field]))
        focused = state["quality"] == "focus"
        for key in ("shadow_focus_label", "shadow_focus_radius_spin", "shadow_refocus_btn"):
            self.widgets[key].setEnabled(focused)
        self.widgets["shadow_bias_spin"].setEnabled(state["quality"] != "standard")
        self._update_shadow_summary(state)

    def _refresh_shadow_summary(self) -> None:
        """Refresh visible coverage metadata after scene loads or animation."""
        group = self.widgets.get("shadow_detail_group")
        renderer = getattr(self.parent, "renderer", None)
        if group is None or not group.isVisible():
            return
        if renderer is not None and renderer_capabilities(renderer).shadow_quality:
            self._update_shadow_summary(renderer.get_shadow_quality_state())

    def _update_shadow_summary(self, state: dict) -> None:
        label = self.widgets["shadow_detail_summary"]
        if state["width"] <= 0 or state["height"] <= 0:
            label.setText("Native shadow map: 1024 × 1024 · waiting for renderer")
            return
        focused = state["quality"] == "focus"
        detail = (
            f"1024 × 1024 · {state['world_units_per_texel']:.3g} m/texel "
            f"· {state['width']:.3g} × {state['height']:.3g} m coverage"
        )
        if focused:
            detail += "\nPinned region only; shadows outside it are not guaranteed."
        elif state["quality"] != "standard" and not state["fit_available"]:
            detail += "\nWaiting for scene meshes."
        label.setText(detail)

    def _on_shadow_quality_changed(self, _index: int) -> None:
        renderer = getattr(self.parent, "renderer", None)
        if renderer is None or not renderer_capabilities(renderer).shadow_quality:
            return
        renderer.set_shadow_quality(self.widgets["shadow_quality_combo"].currentData())
        self._sync_shadow_quality_widgets(renderer)

    def _on_shadow_focus_radius_changed(self, value: float) -> None:
        renderer = getattr(self.parent, "renderer", None)
        if renderer is not None and renderer_capabilities(renderer).shadow_quality:
            renderer.set_shadow_focus_radius(value)
            self._sync_shadow_quality_widgets(renderer)

    def _on_shadow_refocus(self) -> None:
        renderer = getattr(self.parent, "renderer", None)
        if renderer is not None and renderer_capabilities(renderer).shadow_quality:
            renderer.refocus_shadows()
            self._sync_shadow_quality_widgets(renderer)

    def _on_shadow_bias_changed(self, value: float) -> None:
        renderer = getattr(self.parent, "renderer", None)
        if renderer is not None and renderer_capabilities(renderer).shadow_quality:
            renderer.set_shadow_bias_texels(value)
            self._sync_shadow_quality_widgets(renderer)

    def _sync_lighting_profile_widgets(self, renderer: object) -> None:
        """Mirror pygfx's active lighting profile without applying it."""
        combo = self.widgets.get("lighting_profile_combo")
        if combo is None:
            return
        if not renderer_capabilities(renderer).lighting_profiles:
            return
        try:
            profiles = renderer.get_available_lighting_profiles()
            active = renderer.get_lighting_profile()
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return
        if not isinstance(profiles, dict):
            return

        self._syncing_lighting_profile = True
        try:
            with QSignalBlocker(combo):
                combo.clear()
                for key, label in profiles.items():
                    combo.addItem(str(label), str(key))
                index = combo.findData(str(active))
                if index >= 0:
                    combo.setCurrentIndex(index)
        finally:
            self._syncing_lighting_profile = False

    # Paper mode

    def _renderer_supports_wireframe(self) -> bool:
        """Return whether the active renderer exposes the wireframe protocol hook."""
        viz = self.parent
        renderer = getattr(viz, "renderer", None) if viz is not None else None
        return renderer is not None and renderer_capabilities(renderer).wireframe

    def _on_paper_mode_toggled(self, checked: bool) -> None:
        """Route paper mode toggle to activate or deactivate."""
        appearance = getattr(self.parent, "appearance_preset_service", None)
        if appearance is not None:
            if checked:
                appearance.apply("Paper")
            elif appearance.previous is not None:
                appearance.restore(appearance.previous)
            return
        if checked:
            self._activate_paper_mode()
        else:
            self._deactivate_paper_mode()

    def _activate_paper_mode(self) -> None:
        """Snapshot current widget values and apply paper mode defaults."""
        defaults = self._PAPER_MODE_PRESET
        self._paper_mode_saved = self._snapshot_settings(defaults)
        self._apply_settings_to_widgets(defaults)

    def _deactivate_paper_mode(self) -> None:
        """Restore saved widget values from before paper mode."""
        if self._paper_mode_saved is not None:
            self._apply_settings_to_widgets(self._paper_mode_saved)
        self._paper_mode_saved = None

    def _snapshot_settings(self, keys: Dict) -> Dict:
        """Read current widget values for the keys in *keys*."""
        snap: Dict = {}
        for key in keys:
            if key == "bg_combo":
                combo = self.widgets.get("bg_combo")
                if combo is not None:
                    snap[key] = combo.currentText()
            elif key == "shader_combo":
                combo = self.widgets.get("shader_combo")
                if combo is not None:
                    snap[key] = combo.currentText()
            elif key in ("skybox_cb", "outline_cb", "shadows_cb"):
                cb = self.widgets.get(key)
                if cb is not None:
                    snap[key] = cb.isChecked()
            elif key == "ibl_intensity":
                spin = self.widgets.get("ibl_intensity_spin")
                if spin is not None:
                    snap[key] = spin.value()
        return snap

    def _apply_settings_to_widgets(self, settings: Dict) -> None:
        """Batch-update widgets and push values to the renderer.

        Blocks all affected signals first, sets values, unblocks, then pushes
        to the renderer in one pass.
        """
        # Collect widgets whose signals need blocking
        affected = []
        for key in settings:
            if key == "bg_combo":
                affected.append(self.widgets.get("bg_combo"))
            elif key == "shader_combo":
                affected.append(self.widgets.get("shader_combo"))
            elif key in ("skybox_cb", "outline_cb", "shadows_cb"):
                affected.append(self.widgets.get(key))
            elif key == "ibl_intensity":
                affected.append(self.widgets.get("ibl_intensity_slider"))
                affected.append(self.widgets.get("ibl_intensity_spin"))
        affected = [w for w in affected if w is not None]

        blockers = [QSignalBlocker(w) for w in affected]
        try:
            for key, val in settings.items():
                if key == "bg_combo":
                    combo = self.widgets.get("bg_combo")
                    if combo is not None:
                        combo.setCurrentText(val)
                elif key == "shader_combo":
                    combo = self.widgets.get("shader_combo")
                    if combo is not None:
                        combo.setCurrentText(val)
                elif key == "skybox_cb":
                    self._sync_background_controls(bool(val))
                elif key in ("outline_cb", "shadows_cb"):
                    cb = self.widgets.get(key)
                    if cb is not None:
                        cb.setChecked(val)
                elif key == "ibl_intensity":
                    self._sync_ibl_widgets(int(val))
        finally:
            blockers.clear()

        # Push to renderer
        self._apply_settings_to_renderer(settings)

    def _apply_settings_to_renderer(self, settings: Dict) -> None:
        """Push settings values to the renderer/visualizer."""
        viz = self.parent
        if viz is None:
            return
        renderer = getattr(viz, "renderer", None)
        capabilities = renderer_capabilities(renderer)

        for key, val in settings.items():
            if key == "bg_combo":
                color = self._BG_PRESETS.get(val)
                if color is not None:
                    viz.scene_appearance_service.set_background_preset(val, color)
            elif key == "shader_combo":
                if renderer is not None and capabilities.scene_shader:
                    renderer.set_scene_shader(val.lower())
            elif key == "skybox_cb":
                if renderer is not None and capabilities.skybox:
                    renderer.show_skybox(val)
            elif key == "ibl_intensity":
                if renderer is not None and capabilities.ibl:
                    renderer.set_ibl_intensity(float(val))
            elif key == "shadows_cb":
                if renderer is not None and capabilities.shadow_toggle:
                    renderer.set_shadow_enabled(bool(val))
            elif key == "outline_cb":
                scene_appearance = getattr(viz, "scene_appearance_service", None)
                if scene_appearance is not None:
                    scene_appearance.set_edge_visibility(bool(val))

    def _break_paper_mode(self) -> None:
        """Auto-uncheck paper mode when user manually changes a controlled setting."""
        cb = self.widgets.get("paper_mode_cb")
        if cb is not None and cb.isChecked():
            with QSignalBlocker(cb):
                cb.setChecked(False)
            self._paper_mode_saved = None

    # Appearance callbacks

    def _on_background_changed(self, preset_name: str) -> None:
        """Apply a named background preset through its appearance owner."""
        self._break_paper_mode()
        color = self._BG_PRESETS.get(preset_name)
        if color is None:
            return
        self.parent.scene_appearance_service.set_background_preset(preset_name, color)
        self._sync_from_visualizer()

    def _on_skybox_toggled(self, checked: bool) -> None:
        """Select one backdrop while retaining independent environment lighting."""
        self._break_paper_mode()
        self._sync_background_controls(checked)
        renderer = getattr(self.parent, "renderer", None)
        if renderer is not None and renderer_capabilities(renderer).skybox:
            renderer.show_skybox(checked)
            self._sync_from_visualizer()

    def _sync_background_controls(self, skybox_visible: bool | None = None) -> None:
        """Mirror exclusive radios without callbacks, opening IBL on sky selection."""
        skybox = self.widgets.get("skybox_cb")
        solid = self.widgets.get("background_solid_radio")
        if skybox is None or solid is None:
            return
        if skybox_visible is not None:
            with QSignalBlocker(skybox), QSignalBlocker(solid):
                (skybox if skybox_visible else solid).setChecked(True)
        visible = skybox.isChecked()
        self.widgets["background_color_row"].setVisible(not visible)
        # Do not undo a user's manual expansion while editing IBL against a
        # solid backdrop, or repeatedly unfold controls during Demo playback.
        if visible != self._background_skybox_shown:
            environment = self.widgets.get("environment_group")
            if environment is not None:
                (environment.expand if visible else environment.collapse)()
            self._background_skybox_shown = visible

    def _on_shader_changed(self, shader_name: str) -> None:
        """Handle scene shader change."""
        self._break_paper_mode()
        renderer = getattr(self.parent, "renderer", None)
        if renderer is not None and renderer_capabilities(renderer).scene_shader:
            renderer.set_scene_shader(shader_name.lower())

    # Lighting callbacks

    def _sync_ibl_widgets(self, intensity: int) -> None:
        """Sync IBL slider and spin widgets."""
        intensity = max(0, int(intensity))

        slider = self.widgets.get("ibl_intensity_slider")
        spin = self.widgets.get("ibl_intensity_spin")

        if slider is not None:
            with QSignalBlocker(slider):
                slider.setValue(int(intensity / 1000))

        if spin is not None:
            with QSignalBlocker(spin):
                spin.setValue(int(intensity))

    def _on_ibl_slider_changed(self, value: int) -> None:
        """Handle IBL intensity slider change."""
        self._break_paper_mode()
        intensity = value * 1000  # Slider is 0-100, intensity is 0-100000

        # Sync spin
        spin = self.widgets.get("ibl_intensity_spin")
        if spin is not None:
            with QSignalBlocker(spin):
                spin.setValue(intensity)

        self._apply_ibl_intensity(intensity)

    def _on_ibl_spin_changed(self, value: int) -> None:
        """Handle IBL intensity spin change."""
        self._break_paper_mode()
        # Sync slider
        slider = self.widgets.get("ibl_intensity_slider")
        if slider is not None:
            with QSignalBlocker(slider):
                slider.setValue(int(value / 1000))

        self._apply_ibl_intensity(value)

    def _apply_ibl_intensity(self, intensity: int) -> None:
        """Apply IBL intensity to renderer."""
        renderer = getattr(self.parent, "renderer", None)
        if renderer is not None and renderer_capabilities(renderer).ibl:
            renderer.set_ibl_intensity(float(intensity))
            self._sync_lighting_profile_widgets(renderer)

    def _on_lighting_profile_changed(self, _index: int) -> None:
        """Apply a named pygfx lighting profile selected by the user."""
        if self._syncing_lighting_profile:
            return
        renderer = getattr(self.parent, "renderer", None)
        combo = self.widgets.get("lighting_profile_combo")
        if (
            renderer is None
            or combo is None
            or not renderer_capabilities(renderer).lighting_profiles
        ):
            return
        profile = combo.currentData()
        if profile in (None, "custom"):
            self._sync_lighting_profile_widgets(renderer)
            return
        if renderer.set_lighting_profile(str(profile)):
            if renderer_capabilities(renderer).ibl:
                intensity = renderer.get_ibl_intensity()
                if intensity is not None:
                    self._sync_ibl_widgets(int(intensity))
            self._sync_light_rig_widgets(renderer)

    def _on_headlight_toggled(self, checked: bool) -> None:
        """Handle camera-follow headlight toggle."""
        renderer = getattr(self.parent, "renderer", None)
        if renderer is not None and renderer_capabilities(renderer).direct_lighting:
            renderer.set_headlight_enabled(bool(checked))
            self._sync_lighting_profile_widgets(renderer)

    def _on_headlight_intensity_changed(self, value: float) -> None:
        """Handle headlight intensity changes."""
        renderer = getattr(self.parent, "renderer", None)
        if renderer is not None and renderer_capabilities(renderer).direct_lighting:
            renderer.set_headlight_intensity(float(value))
            self._sync_lighting_profile_widgets(renderer)

    def _on_key_light_angles_changed(self, _value: float) -> None:
        """Handle key light azimuth/elevation changes."""
        renderer = getattr(self.parent, "renderer", None)
        if renderer is None or not renderer_capabilities(renderer).direct_lighting:
            return
        az_w = self.widgets.get("key_light_az_spin")
        el_w = self.widgets.get("key_light_el_spin")
        if az_w is None or el_w is None:
            return
        renderer.set_key_light_angles(float(az_w.value()), float(el_w.value()))
        self._sync_lighting_profile_widgets(renderer)

    def _on_key_light_intensity_changed(self, value: float) -> None:
        """Handle key light intensity changes."""
        renderer = getattr(self.parent, "renderer", None)
        if renderer is not None and renderer_capabilities(renderer).direct_lighting:
            renderer.set_key_light_intensity(float(value))
            self._sync_lighting_profile_widgets(renderer)

    def _on_fill_light_angles_changed(self, _value: float) -> None:
        """Handle fill light azimuth/elevation changes."""
        renderer = getattr(self.parent, "renderer", None)
        if renderer is None or not renderer_capabilities(renderer).direct_lighting:
            return
        az_w = self.widgets.get("fill_light_az_spin")
        el_w = self.widgets.get("fill_light_el_spin")
        if az_w is None or el_w is None:
            return
        renderer.set_fill_light_angles(float(az_w.value()), float(el_w.value()))
        self._sync_lighting_profile_widgets(renderer)

    def _on_fill_light_intensity_changed(self, value: float) -> None:
        """Handle fill light intensity changes."""
        renderer = getattr(self.parent, "renderer", None)
        if renderer is not None and renderer_capabilities(renderer).direct_lighting:
            renderer.set_fill_light_intensity(float(value))
            self._sync_lighting_profile_widgets(renderer)

    def _on_ibl_changed(self, name: str) -> None:
        """Select an installed map or offer an explicit optional-sky download."""
        if not name:
            return
        path = self._ibl_library.get(name)
        if not path:
            return
        renderer = getattr(self.parent, "renderer", None)
        if renderer is None or not renderer_capabilities(renderer).ibl:
            return

        download = self.widgets.get("environment_download")
        if renderer_capabilities(renderer).hdr_ibl:
            if download is not None:
                download.select(str(path))
            if path in environment_catalog():
                return
            if not renderer.set_ibl(str(path)):
                logger.warning("Could not load HDR environment '%s'", path)
                self._sync_ibl_selection(renderer.get_ibl_name())
            self._sync_from_visualizer()
            return

        path_str = str(path)
        path_lower = path_str.lower()

        # Handle "default" special case
        if path_lower == "default":
            renderer.set_ibl("default")
            self._sync_from_visualizer()
            return

        # Reject non-KTX files (HDR, EXR, PNG, JPG, etc.)
        if not path_lower.endswith(".ktx"):
            logger.warning(
                "IBL '%s' is not a KTX file. Use pre-converted IBL pairs "
                "(name_ibl.ktx + name_skybox.ktx). See libraries/ibl/README.md.",
                path_str,
            )
            return

        # Handle _skybox.ktx -> find matching _ibl.ktx
        if path_lower.endswith("_skybox.ktx"):
            name_base = Path(path_str).name[:-11]
            ibl_path = Path(path_str).with_name(f"{name_base}_ibl.ktx")
            if ibl_path.exists():
                path_str = str(ibl_path)
            else:
                logger.warning("IBL skybox '%s' has no matching _ibl.ktx", path_str)
                return

        # Validate _ibl.ktx naming
        if not path_str.lower().endswith("_ibl.ktx"):
            logger.warning(
                "IBL '%s' must end with _ibl.ktx and be paired with _skybox.ktx",
                path_str,
            )
            return

        renderer.set_ibl(path_str)
        self._sync_from_visualizer()

        skybox_cb = self.widgets.get("skybox_cb")
        if skybox_cb is not None and skybox_cb.isChecked():
            if renderer_capabilities(renderer).skybox:
                renderer.show_skybox(True)

    def _on_load_ibl(self) -> None:
        """Load a custom KTX IBL environment.

        pygfx accepts Radiance HDR directly; Open3D uses a matching KTX pair.
        """
        renderer = getattr(self.parent, "renderer", None)
        if renderer_capabilities(renderer).hdr_ibl:
            file_path, _ = QFileDialog.getOpenFileName(
                self.parent, "Load HDR Environment", "", "Radiance HDR (*.hdr *.HDR)"
            )
            if not file_path:
                return
            display_name = self._make_unique_ibl_name(Path(file_path).stem)
            self._ibl_library[display_name] = file_path
            combo = self.widgets.get("ibl_combo")
            if combo is not None:
                with QSignalBlocker(combo):
                    combo.addItem(display_name)
                    combo.setCurrentText(display_name)
            self._on_ibl_changed(display_name)
            return
        start_dir = ""
        if self._ibl_dir is not None and self._ibl_dir.exists():
            start_dir = str(self._ibl_dir)
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent,
            "Load IBL Environment",
            start_dir,
            "KTX IBL Files (*_ibl.ktx);;All KTX Files (*.ktx);;All Files (*)",
        )
        if not file_path:
            return

        # Validate it's a proper _ibl.ktx file
        if not file_path.lower().endswith("_ibl.ktx"):
            logger.warning("IBL file must end with _ibl.ktx. Use cmgen to convert HDR files.")
            return

        skybox_path = file_path.replace("_ibl.ktx", "_skybox.ktx")
        if not Path(skybox_path).exists():
            logger.warning("IBL '%s' has no matching _skybox.ktx file", file_path)
            return

        base_name = Path(file_path).name[:-8] or "Custom"  # Remove "_ibl.ktx"
        display_name = self._make_unique_ibl_name(base_name)
        self._ibl_library[display_name] = file_path
        combo = self.widgets.get("ibl_combo")
        if combo is not None:
            with QSignalBlocker(combo):
                combo.addItem(display_name)
                combo.setCurrentText(display_name)
        self._on_ibl_changed(display_name)

    def _sync_ibl_selection(self, ibl_name: str) -> None:
        """Reflect renderer-selected IBL names or paths in the combo box."""
        combo = self.widgets.get("ibl_combo")
        if combo is None or not ibl_name:
            return
        download = self.widgets.get("environment_download")
        if download is not None and download.awaiting_download:
            return  # Keep the pending download visible while the previous sky stays active.
        display_name = self._find_ibl_display_name(ibl_name)
        if display_name is None:
            base_name = Path(ibl_name).stem or "Custom"
            display_name = self._make_unique_ibl_name(base_name)
            self._ibl_library[display_name] = ibl_name
            with QSignalBlocker(combo):
                combo.addItem(display_name)
        with QSignalBlocker(combo):
            combo.setCurrentText(display_name)

    def _find_ibl_display_name(self, ibl_name: str) -> Optional[str]:
        """Return the combo label registered for one renderer IBL identifier."""
        for name, path in self._ibl_library.items():
            if (
                self._hdr_ibl_mode
                and name == "Default"
                and ibl_name in ("default", "neutral_outdoor")
            ):
                return name
            if path == ibl_name:
                return name
        return None

    def _make_unique_ibl_name(self, base_name: str) -> str:
        """Generate a non-conflicting display name for a custom IBL entry."""
        name = base_name
        while name in self._ibl_library:
            self._ibl_custom_index += 1
            name = f"{base_name} ({self._ibl_custom_index})"
        return name

    def _load_ibl_library_from_disk(self) -> None:
        """Load KTX IBL files from the libraries/ibl directory.

        Only loads _ibl.ktx files that have matching _skybox.ktx pairs.
        HDR/EXR files are ignored - they should be pre-converted to KTX using cmgen.
        """
        project_root = self._project_root
        paths = load_app_paths(project_root)
        ibl_rel = paths.get("ibl", "libraries/ibl")
        ibl_dir = (project_root / ibl_rel).resolve()
        self._ibl_dir = ibl_dir

        if self._hdr_ibl_mode:
            self._ibl_library = {"Default": "neutral_outdoor"}
            self._ibl_library.update(
                {asset.label: asset.id for asset in environment_catalog().values()}
            )
            return

        if not ibl_dir.exists():
            return

        # Only load _ibl.ktx files that have matching _skybox.ktx
        for entry in sorted(ibl_dir.iterdir()):
            if not entry.is_file():
                continue
            if not entry.name.endswith("_ibl.ktx"):
                continue
            skybox_path = entry.with_name(entry.name.replace("_ibl.ktx", "_skybox.ktx"))
            if not skybox_path.exists():
                logger.warning("IBL '%s' has no matching skybox file", entry)
                continue
            # Create display name from the base name (strip _ibl.ktx and _4k suffix)
            base_name = entry.name[:-8]  # Remove "_ibl.ktx"
            display_name = self._make_unique_ibl_name(base_name)
            self._ibl_library[display_name] = str(entry)

    def _refresh_ibl_library(self, renderer: object) -> None:
        """Keep pygfx's built-in list small and independent of legacy KTX/LFS files."""
        hdr = renderer_capabilities(renderer).hdr_ibl
        if hdr != self._hdr_ibl_mode:
            download = self.widgets.get("environment_download")
            if download is not None:
                download.select("")
            self._hdr_ibl_mode = hdr
            self._ibl_library = {"Default": "default"}
            self._load_ibl_library_from_disk()
            combo = self.widgets.get("ibl_combo")
            if combo is not None:
                with QSignalBlocker(combo):
                    combo.clear()
                    combo.addItems(self._ibl_library)
        button = self.widgets.get("ibl_load_btn")
        if button is not None:
            button.setText("Load HDR…" if hdr else "Load KTX…")
            button.setToolTip(
                "Load a local Radiance .hdr image" if hdr else "Load a matching KTX IBL/skybox pair"
            )

    def _apply_optional_environment(self, identifier: str) -> None:
        """Called on the GUI thread after an explicit selection or verified download."""
        if getattr(self.parent, "_shutdown_started", False):
            return
        renderer = getattr(self.parent, "renderer", None)
        if not renderer_capabilities(renderer).hdr_ibl:
            return
        download = self.widgets["environment_download"]
        if renderer.set_ibl(identifier):
            if download.match_lighting.isChecked():
                renderer.apply_environment_lighting(identifier)
            skybox = self.widgets.get("skybox_cb")
            if skybox is not None:
                renderer.show_skybox(skybox.isChecked())
            self._sync_ibl_widgets(int(renderer.get_ibl_intensity()))
            self._sync_light_rig_widgets(renderer)
            self._sync_ibl_selection(identifier)
            self._sync_from_visualizer()
        else:
            download.status.setText(
                "Could not load this HDR. Current lighting is unchanged; select Default or retry."
            )

    def cleanup(self) -> None:
        """Cancel pending environment I/O before renderer and Qt resources close."""
        download = self.widgets.get("environment_download")
        if download is not None:
            download.cleanup()

    def _on_shadows_toggled(self, checked: bool) -> None:
        """Handle shadow enable/disable toggle."""
        self._break_paper_mode()
        renderer = getattr(self.parent, "renderer", None)
        if renderer is None or not renderer_capabilities(renderer).shadow_toggle:
            return
        renderer.set_shadow_enabled(bool(checked))
        self._sync_lighting_profile_widgets(renderer)

    def _on_culling_toggled(self, checked: bool) -> None:
        """Handle culling enable/disable toggle."""
        renderer = getattr(self.parent, "renderer", None)
        if (
            renderer is not None
            and renderer_capabilities(renderer).frustum_culling
            and hasattr(renderer, "set_culling")
        ):
            renderer.set_culling(checked)

    def _on_o3d_settings_toggled(self, checked: bool) -> None:
        """Show/hide Open3D's built-in settings panel."""
        renderer = getattr(self.parent, "renderer", None)
        if renderer is not None and renderer_capabilities(renderer).open3d_settings_panel:
            renderer.show_settings(checked)

    # Transparency callbacks (with debounce)

    def _on_building_alpha_slider_changed(self, value: int) -> None:
        """Handle building transparency slider change with debounce."""
        self._break_paper_mode()
        alpha = value / 100.0

        # Debounce the actual application
        self._pending_building_alpha = alpha
        if self._building_alpha_timer is None:
            self._building_alpha_timer = QTimer()
            self._building_alpha_timer.setSingleShot(True)
            self._building_alpha_timer.timeout.connect(self._apply_building_alpha)
        self._building_alpha_timer.start(SLIDER_DEBOUNCE_MS)

    def _apply_building_alpha(self) -> None:
        """Apply debounced building alpha."""
        if self._pending_building_alpha is not None:
            self.parent.scene_appearance_service.set_building_transparency(
                self._pending_building_alpha
            )
            self._pending_building_alpha = None

    def _on_target_alpha_slider_changed(self, value: int) -> None:
        """Handle target transparency slider change with debounce."""
        self._break_paper_mode()
        alpha = value / 100.0

        # Debounce the actual application
        self._pending_target_alpha = alpha
        if self._target_alpha_timer is None:
            self._target_alpha_timer = QTimer()
            self._target_alpha_timer.setSingleShot(True)
            self._target_alpha_timer.timeout.connect(self._apply_target_alpha)
        self._target_alpha_timer.start(SLIDER_DEBOUNCE_MS)

    def _apply_target_alpha(self) -> None:
        """Apply debounced target alpha."""
        if self._pending_target_alpha is not None:
            self.parent.scene_appearance_service.set_target_transparency(self._pending_target_alpha)
            self._pending_target_alpha = None

    # Scene elements callbacks

    def _on_outline_toggled(self, checked: bool) -> None:
        """Toggle scene mesh edge visibility through the appearance service."""
        self._break_paper_mode()
        scene_appearance = getattr(self.parent, "scene_appearance_service", None)
        if scene_appearance is not None:
            scene_appearance.set_edge_visibility(bool(checked))

    def _on_target_outline_toggled(self, checked: bool) -> None:
        """Toggle target mesh edge visibility through the target owner."""
        self._break_paper_mode()
        self.parent.target_service.set_target_edge_visibility(bool(checked))

    def _on_wireframe_toggled(self, checked: bool) -> None:
        """Toggle renderer-level wireframe only when the capability is present."""
        self._break_paper_mode()
        renderer = getattr(self.parent, "renderer", None)
        if (
            self._renderer_supports_wireframe()
            and renderer is not None
            and hasattr(renderer, "set_wireframe")
        ):
            renderer.set_wireframe(bool(checked))

    def _on_show_axes_toggled(self, checked: bool) -> None:
        """Handle show axes toggle."""
        self._break_paper_mode()
        renderer = getattr(self.parent, "renderer", None)
        if renderer is not None and renderer_capabilities(renderer).axes:
            renderer.show_axes(checked)

    def _on_screen_space_labels_toggled(self, checked: bool) -> None:
        """Rebuild TX/RX labels to reflect the new label sizing mode."""
        viz = self.parent
        renderer = getattr(viz, "renderer", None)
        if not bool(renderer is not None and renderer_capabilities(renderer).screen_space_labels):
            return

        app_state = getattr(viz, "app_state", None)
        if app_state is not None and hasattr(app_state, "label_screen_space"):
            viz.set_state(label_screen_space=bool(checked))

        node_service = getattr(viz, "node_service", None)
        if node_service is not None and hasattr(node_service, "recreate_tx_rx_labels"):
            font_size = float(getattr(viz, "label_font_size", 0.3))
            try:
                node_service.recreate_tx_rx_labels(font_size)
                if hasattr(node_service, "recreate_target_labels"):
                    node_service.recreate_target_labels(font_size)
            except Exception as exc:
                logger = getattr(viz, "logger", None)
                if logger is not None:
                    logger.debug("Failed to rebuild TX/RX labels: %s", exc)

        if hasattr(viz, "renderer") and viz.renderer is not None:
            update_fn = getattr(viz.renderer, "update_renderer", None)
            if callable(update_fn):
                try:
                    update_fn()
                except Exception:
                    pass

    def _on_hover_info_mode_changed(self) -> None:
        """Persist and apply the renderer hover tooltip policy."""
        combo = self.widgets.get("hover_info_combo")
        if combo is None:
            return
        mode = combo.currentData()
        if mode is None:
            mode = str(combo.currentText()).strip().lower().replace(" ", "_")

        viz = self.parent
        app_state = getattr(viz, "app_state", None)
        if app_state is not None:
            viz.set_state(hover_info_mode=str(mode))

        renderer = getattr(viz, "renderer", None)
        if renderer is not None and renderer_capabilities(renderer).hover_info:
            try:
                renderer.set_hover_info_mode(str(mode))
            except Exception as exc:
                logger.debug("Failed to set hover info mode: %s", exc)

    def _sync_viewport_hud_controls(self) -> None:
        """Mirror the AppState HUD policy into controls without emitting intent."""
        state = getattr(self.parent, "app_state", None)
        if state is None:
            return
        mode = normalize_viewport_hud_mode(getattr(state, "viewport_hud_mode", "compact"))
        combo = self.widgets.get("viewport_hud_mode_combo")
        if combo is not None:
            index = combo.findData(mode)
            if index >= 0:
                with QSignalBlocker(combo):
                    combo.setCurrentIndex(index)

        style = normalize_viewport_hud_style(getattr(state, "viewport_hud_style", "classic"))
        style_combo = self.widgets.get("viewport_hud_style_combo")
        if style_combo is not None:
            index = style_combo.findData(style)
            if index >= 0:
                with QSignalBlocker(style_combo):
                    style_combo.setCurrentIndex(index)

        for key in ("status", "legends", "filters", "annotations"):
            checkbox = self.widgets.get(f"viewport_hud_{key}_cb")
            if checkbox is None:
                continue
            checked = bool(getattr(state, f"viewport_hud_show_{key}", True))
            with QSignalBlocker(checkbox):
                checkbox.setChecked(checked)
            # The persistent Context switch controls live visibility only.
            # Keep these saved content choices editable while the HUD is off.
            checkbox.setEnabled(True)

    def _apply_viewport_hud_changes(self, **changes: Any) -> None:
        """Persist HUD changes and ask the active renderer to refresh once."""
        viz = self.parent
        if viz is None or getattr(viz, "app_state", None) is None:
            return
        viz.set_state(**changes)
        renderer = getattr(viz, "renderer", None)
        if (
            renderer is not None
            and renderer_capabilities(renderer).viewport_hud
            and hasattr(renderer, "refresh_viewport_hud")
        ):
            renderer.refresh_viewport_hud()
        self._sync_viewport_hud_controls()

    def _on_viewport_hud_mode_changed(self, _index: int) -> None:
        """Apply the selected Compact/Detailed HUD content mode."""
        combo = self.widgets.get("viewport_hud_mode_combo")
        if combo is None:
            return
        mode = normalize_viewport_hud_mode(combo.currentData())
        self._apply_viewport_hud_changes(viewport_hud_mode=mode)

    def _on_viewport_hud_style_changed(self, _index: int) -> None:
        """Apply the selected HUD appearance independently of content and visibility."""
        combo = self.widgets.get("viewport_hud_style_combo")
        if combo is None:
            return
        style = normalize_viewport_hud_style(combo.currentData())
        self._apply_viewport_hud_changes(viewport_hud_style=style)

    def _on_viewport_hud_category_changed(self, key: str, checked: bool) -> None:
        """Apply one independent viewport-HUD content category."""
        if key not in {"status", "legends", "filters", "annotations"}:
            return
        self._apply_viewport_hud_changes(**{f"viewport_hud_show_{key}": bool(checked)})

    def apply_runtime_controls_from_state(self) -> None:
        """Apply AppState-backed render controls to the live renderer/UI."""
        viz = self.parent
        if viz is None:
            return

        app_state = getattr(viz, "app_state", None)
        renderer = getattr(viz, "renderer", None)
        if app_state is None:
            return

        if renderer is not None and renderer_capabilities(renderer).antialiasing:
            renderer.set_antialiasing_mode(str(getattr(app_state, "antialiasing_mode", "off")))
        depth_precision = getattr(app_state, "depth_precision", None)
        if (
            depth_precision is not None
            and renderer is not None
            and renderer_capabilities(renderer).depth_precision
        ):
            renderer.set_depth_precision_enabled(bool(depth_precision))
        mipmaps = getattr(app_state, "texture_mipmaps", None)
        if (
            mipmaps is not None
            and renderer is not None
            and renderer_capabilities(renderer).texture_mipmaps
        ):
            renderer.set_texture_mipmaps_enabled(bool(mipmaps))
        if renderer is not None and renderer_capabilities(renderer).ground_grid:
            renderer.set_ground_grid_visible(bool(getattr(app_state, "show_ground_grid", False)))
        if renderer is not None and renderer_capabilities(renderer).hover_info:
            renderer.set_hover_info_mode(str(getattr(app_state, "hover_info_mode", "essential")))
        if (
            renderer is not None
            and renderer_capabilities(renderer).viewport_hud
            and hasattr(renderer, "refresh_viewport_hud")
        ):
            renderer.refresh_viewport_hud()
        self._push_clipping_planes_to_renderer()

        if renderer is not None and hasattr(renderer, "refresh_mpc_point_markers"):
            try:
                renderer.refresh_mpc_point_markers()
            except Exception as exc:
                logger.debug("Failed to refresh MPC markers from state: %s", exc)

        node_service = getattr(viz, "node_service", None)
        if node_service is not None and hasattr(node_service, "recreate_tx_rx_labels"):
            font_size = float(getattr(viz, "label_font_size", 0.3))
            try:
                node_service.recreate_tx_rx_labels(font_size)
            except Exception as exc:
                logger.debug("Failed to rebuild TX/RX labels from state: %s", exc)

    # Cutaway / clipping plane handlers (pygfx only)

    def _push_clipping_planes_to_renderer(self) -> None:
        """Compute active clipping planes from AppState and send to renderer.

        Plane tuples follow pygfx's ``(nx, ny, nz, d)`` convention: fragments
        are kept where ``n·world_pos >= d``. ``flip=False`` keeps the negative
        half-space (e.g. ``z <= position``) — this matches the slider's
        intuitive "cut off the top half" feel when the user drags rightward.
        """
        viz = self.parent
        app_state = getattr(viz, "app_state", None)
        renderer = getattr(viz, "renderer", None)
        if renderer is None or not renderer_capabilities(renderer).clipping_planes:
            return

        planes: list[tuple[float, float, float, float]] = []
        if app_state is not None:
            axes = (
                ("x", (1.0, 0.0, 0.0)),
                ("y", (0.0, 1.0, 0.0)),
                ("z", (0.0, 0.0, 1.0)),
            )
            for axis, normal in axes:
                if not getattr(app_state, f"clip_{axis}_enabled", False):
                    continue
                pos = float(getattr(app_state, f"clip_{axis}_position", 0.0))
                flip = bool(getattr(app_state, f"clip_{axis}_flip", False))
                sign = 1.0 if flip else -1.0
                nx, ny, nz = (sign * c for c in normal)
                planes.append((nx, ny, nz, sign * pos))

        try:
            renderer.set_clipping_planes(tuple(planes))
        except Exception as exc:
            logger.debug("Failed to apply clipping planes: %s", exc)

    def _update_clip_app_state(self, axis: str, **changes: object) -> None:
        """Write one axis of clipping-plane state back into AppState."""
        viz = self.parent
        app_state = getattr(viz, "app_state", None)
        if app_state is None:
            return
        prefixed = {f"clip_{axis}_{key}": value for key, value in changes.items()}
        viz.set_state(**prefixed)

    def _on_clip_enabled_toggled(self, axis: str, checked: bool) -> None:
        """Enable or disable one clipping plane and push the renderer update."""
        self._break_paper_mode()
        self._update_clip_app_state(axis, enabled=bool(checked))
        self._push_clipping_planes_to_renderer()

    def _on_clip_slider_changed(self, axis: str, value: int) -> None:
        """Mirror slider changes into the meter-valued spinbox and AppState."""
        self._break_paper_mode()
        position = float(value)
        spin = self.widgets.get(f"clip_{axis}_spin")
        if spin is not None:
            with QSignalBlocker(spin):
                spin.setValue(position)
        self._update_clip_app_state(axis, position=position)
        self._push_clipping_planes_to_renderer()

    def _on_clip_spin_changed(self, axis: str, value: float) -> None:
        """Mirror spinbox meter values into the slider and AppState."""
        self._break_paper_mode()
        slider = self.widgets.get(f"clip_{axis}_slider")
        if slider is not None:
            clamped = int(max(slider.minimum(), min(slider.maximum(), round(value))))
            with QSignalBlocker(slider):
                slider.setValue(clamped)
        self._update_clip_app_state(axis, position=float(value))
        self._push_clipping_planes_to_renderer()

    def _on_clip_flip_toggled(self, axis: str, checked: bool) -> None:
        """Flip which half-space a clipping plane keeps."""
        self._break_paper_mode()
        self._update_clip_app_state(axis, flip=bool(checked))
        self._push_clipping_planes_to_renderer()

    # Live image quality and scene-guide handlers

    def _on_aa_mode_changed(self, label: str) -> None:
        """Persist and apply the selected interactive anti-aliasing mode."""
        self._break_paper_mode()
        mode = str(label).lower()
        viz = self.parent
        app_state = getattr(viz, "app_state", None)
        if app_state is not None:
            viz.set_state(antialiasing_mode=mode)
        renderer = getattr(viz, "renderer", None)
        if renderer is not None and renderer_capabilities(renderer).antialiasing:
            try:
                renderer.set_antialiasing_mode(mode)
            except Exception as exc:
                logger.debug("Failed to set AA mode: %s", exc)

    def _on_depth_precision_toggled(self, checked: bool) -> None:
        """Apply camera precision without changing geometry or scene data."""
        viz = self.parent
        renderer = getattr(viz, "renderer", None)
        if renderer is None or not renderer_capabilities(renderer).depth_precision:
            return
        renderer.set_depth_precision_enabled(bool(checked))
        if getattr(viz, "app_state", None) is not None:
            viz.set_state(depth_precision=bool(checked))

    def _on_texture_mipmaps_toggled(self, checked: bool) -> None:
        """Apply filtering immediately and retain the explicit workspace choice."""
        viz = self.parent
        renderer = getattr(viz, "renderer", None)
        if renderer is None or not renderer_capabilities(renderer).texture_mipmaps:
            return
        renderer.set_texture_mipmaps_enabled(bool(checked))
        if getattr(viz, "app_state", None) is not None:
            viz.set_state(texture_mipmaps=bool(checked))

    def _on_ground_grid_toggled(self, checked: bool) -> None:
        """Persist and apply the renderer ground-grid visibility."""
        self._break_paper_mode()
        viz = self.parent
        app_state = getattr(viz, "app_state", None)
        if app_state is not None:
            viz.set_state(show_ground_grid=bool(checked))
        renderer = getattr(viz, "renderer", None)
        if renderer is not None and renderer_capabilities(renderer).ground_grid:
            try:
                renderer.set_ground_grid_visible(bool(checked))
            except Exception as exc:
                logger.debug("Failed to toggle ground grid: %s", exc)

    def _on_clip_reset_clicked(self) -> None:
        """Disable all clipping planes and reset flip state."""
        self._break_paper_mode()
        for axis in ("x", "y", "z"):
            cb = self.widgets.get(f"clip_{axis}_cb")
            if cb is not None:
                with QSignalBlocker(cb):
                    cb.setChecked(False)
            flip = self.widgets.get(f"clip_{axis}_flip")
            if flip is not None:
                with QSignalBlocker(flip):
                    flip.setChecked(False)
            self._update_clip_app_state(axis, enabled=False, flip=False)
        self._push_clipping_planes_to_renderer()

    def _on_edge_line_width_slider_changed(self, value: int) -> None:
        """Handle edge line width slider change."""
        self._break_paper_mode()
        spin = self.widgets.get("edge_line_width_spin")
        if spin is not None:
            with QSignalBlocker(spin):
                spin.setValue(value)

        renderer = getattr(self.parent, "renderer", None)
        if (
            self._renderer_supports_wireframe()
            and renderer is not None
            and hasattr(renderer, "set_edge_line_width")
        ):
            renderer.set_edge_line_width(float(value))

    def _on_edge_line_width_spin_changed(self, value: int) -> None:
        """Handle edge line width spin change."""
        self._break_paper_mode()
        slider = self.widgets.get("edge_line_width_slider")
        if slider is not None:
            with QSignalBlocker(slider):
                slider.setValue(value)

        renderer = getattr(self.parent, "renderer", None)
        if (
            self._renderer_supports_wireframe()
            and renderer is not None
            and hasattr(renderer, "set_edge_line_width")
        ):
            renderer.set_edge_line_width(float(value))
