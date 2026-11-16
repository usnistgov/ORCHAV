"""Layout composition for the Scenario Builder workspace."""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

from generator.core.materials.target_materials import available_target_material_types
from shared.scenarios.actors import (
    MAX_RANDOM_SEED,
)

from ..widgets.form_layout import AlignedFormLayout
from .domain import (
    ActorRole,
    QualityPreset,
)
from .inspector_scroll import InspectorScrollArea
from .mobility_editor import MobilityEditor
from .orientation_editor import OrientationEditor
from .osm_panel import OsmScenePanel
from .target_representation_editor import TargetRepresentationEditor
from .viewport import ScenarioAuthoringViewport
from .viewport_port import (
    AuthoringTool,
    OverlayVisibility,
)


class AuthoringWorkspaceUI:
    """Compose the viewport, contextual inspector, and optional detail drawer."""

    def _build_ui(
        self,
        viewport_factory: Callable[[Any, QWidget | None], QWidget] | None,
    ) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        header = QHBoxLayout()
        self.document_label = QLabel("Untitled Scenario")
        self.read_only_label = QLabel("READ ONLY")
        self.read_only_label.setVisible(False)
        self.read_only_label.setStyleSheet("font-weight: 700; color: #e0a020;")
        header.addWidget(self.document_label)
        header.addWidget(self.read_only_label)
        header.addStretch()
        header.addWidget(QLabel("Node colours"))
        self.node_color_mode = QComboBox()
        self.node_color_mode.addItem("By role", "role")
        self.node_color_mode.addItem("Node ID", "individual")
        self.node_color_mode.setToolTip(
            "One colour per radio, in transmitter then receiver order; validation warnings retain their colours."
        )
        self.node_color_mode.currentIndexChanged.connect(self._viewport_appearance_changed)
        header.addWidget(self.node_color_mode)
        from ..materials.view_styles import PRESETS

        header.addWidget(QLabel("Appearance"))
        self.appearance_preset = QComboBox()
        self.appearance_preset.addItems([name for name in PRESETS if name != "Plan"])
        appearance_service = getattr(self.visualizer, "appearance_preset_service", None)
        if appearance_service is not None:
            self.appearance_preset.addItems(appearance_service.names())
        self.appearance_preset.currentTextChanged.connect(self._appearance_preset_changed)
        header.addWidget(self.appearance_preset)
        self.leave_authoring_button = QPushButton("Return to Visualization")
        self.leave_authoring_button.clicked.connect(self.leave_authoring_requested.emit)
        header.addWidget(self.leave_authoring_button)
        root.addLayout(header)

        self.recording_banner = QWidget(self)
        recording_layout = QHBoxLayout(self.recording_banner)
        recording_layout.setContentsMargins(4, 2, 4, 2)
        self.recording_explanation = QLabel(
            "Viewing saved results. Create an editable variant to change the acquisition inputs."
        )
        self.recording_explanation.setWordWrap(True)
        recording_layout.addWidget(self.recording_explanation, 1)
        self.inspect_recording_button = QPushButton("Inspect recording")
        self.inspect_recording_button.setToolTip(
            "Open the recording in the main visualizer, including saved sensing results."
        )
        self.inspect_recording_button.clicked.connect(self.inspect_recording_requested.emit)
        recording_layout.addWidget(self.inspect_recording_button)
        self.create_variant_button = QPushButton("Create editable variant…")
        self.create_variant_button.clicked.connect(self.create_variant_requested.emit)
        recording_layout.addWidget(self.create_variant_button)
        self.recording_banner.hide()
        root.addWidget(self.recording_banner)

        body_splitter = QSplitter(Qt.Horizontal, self)
        body_splitter.setObjectName("authoringBodySplitter")
        body_splitter.addWidget(self._build_actor_panel())

        viewport_column = QWidget(body_splitter)
        viewport_layout = QVBoxLayout(viewport_column)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.addWidget(self._build_preview_summary())
        viewport_layout.addWidget(self._build_viewport_toolbar())
        viewport_layout.addWidget(self._build_waypoint_session_bar())
        factory = viewport_factory or ScenarioAuthoringViewport
        self.viewport = factory(self.visualizer, viewport_column)
        self.camera_mode_combo.setEnabled(
            callable(getattr(getattr(self.viewport, "port", None), "set_camera_mode", None))
        )
        self.fit_actors_button.setEnabled(
            callable(getattr(getattr(self.viewport, "port", None), "fit_actors", None))
        )
        self.fit_scene_button.setEnabled(
            callable(getattr(getattr(self.viewport, "port", None), "fit_all", None))
        )
        viewport_layout.addWidget(self.viewport, 1)
        self.coordinate_label = QLabel("X: --   Y: --   Z: --   | Work plane Z=0 m | Snap off")
        viewport_layout.addWidget(self.coordinate_label)
        self.osm_map_credit = QLabel(self.osm_panel.attribution.text())
        self.osm_map_credit.setOpenExternalLinks(True)
        self.osm_map_credit.setWordWrap(True)
        self.osm_map_credit.hide()
        viewport_layout.addWidget(self.osm_map_credit)
        input_signal = getattr(self.viewport, "input_received", None)
        if input_signal is not None:
            input_signal.connect(self._on_viewport_input)
        body_splitter.addWidget(viewport_column)
        self.inspector_splitter = QSplitter(Qt.Horizontal, body_splitter)
        self.inspector_splitter.setObjectName("authoringContextSplitter")
        from .trajectory_preview import TrajectoryPreview

        self.trajectory_preview = TrajectoryPreview(self.inspector_splitter)
        self.trajectory_preview.hide()
        self.inspector_splitter.addWidget(self.trajectory_preview)
        self.inspector_splitter.addWidget(self._build_inspector())
        self.inspector_splitter.setSizes((220, 300))
        body_splitter.addWidget(self.inspector_splitter)
        body_splitter.setStretchFactor(0, 0)
        body_splitter.setStretchFactor(1, 1)
        body_splitter.setStretchFactor(2, 0)
        body_splitter.setSizes((240, 760, 300))
        root.addWidget(body_splitter, 1)

        self.drawer = self._build_drawer()
        self.drawer.setMaximumHeight(240)
        self.drawer.hide()
        self.playback_strip = QWidget(self)
        playback = QHBoxLayout(self.playback_strip)
        playback.setContentsMargins(2, 0, 2, 0)
        playback.addWidget(self.play_button)
        self.play_button.setMaximumWidth(65)
        playback.addWidget(self.timeline_slider, 1)
        self.frame_label = QLabel("Frame 1", self.playback_strip)
        playback.addWidget(self.frame_label)
        for button in (self.save_button, self.save_as_button, self.generate_button):
            playback.addWidget(button)
        self.details_toggle = QPushButton("Details / advanced", self.playback_strip)
        self.details_toggle.setCheckable(True)
        self.details_toggle.toggled.connect(self.drawer.setVisible)
        playback.addWidget(self.details_toggle)
        root.addWidget(self.playback_strip)
        root.addWidget(self.drawer)
        QTimer.singleShot(0, self._appearance_preset_changed)

    def _build_actor_panel(self) -> QWidget:
        panel = QWidget(self)
        self.actor_panel = panel
        panel.setObjectName("authoringActorPanel")
        panel.setMinimumWidth(225)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        scene_group = QGroupBox("Scene", panel)
        scene_form = AlignedFormLayout(scene_group)
        self.scene_source_combo = QComboBox(scene_group)
        for label, source in (
            ("Sionna built-in", "sionna"),
            ("ORCHAV library", "library"),
            ("Local XML", "local"),
            ("OpenStreetMap", "osm"),
        ):
            self.scene_source_combo.addItem(label, source)
        self.scene_source_combo.currentIndexChanged.connect(self._scene_source_changed)
        self.scene_id_combo = QComboBox(scene_group)
        self.scene_id_combo.setEditable(True)
        self.scene_id_combo.setInsertPolicy(QComboBox.NoInsert)
        self.scene_id_combo.setToolTip(
            "Choose a catalog scene, or enter the exact supported scene identifier."
        )
        scene_buttons = QWidget(scene_group)
        scene_button_layout = QHBoxLayout(scene_buttons)
        scene_button_layout.setContentsMargins(0, 0, 0, 0)
        self.scene_browse_button = QPushButton("Browse XML...", scene_buttons)
        self.scene_browse_button.clicked.connect(self._browse_local_scene)
        self.scene_apply_button = QPushButton("Apply Scene", scene_buttons)
        self.scene_apply_button.clicked.connect(self._apply_scene)
        scene_button_layout.addWidget(self.scene_browse_button)
        scene_button_layout.addWidget(self.scene_apply_button)
        scene_form.addRow("Source", self.scene_source_combo)
        scene_form.addRow("ID / path", self.scene_id_combo)
        scene_form.addRow(scene_buttons)
        self._scene_id_label = scene_form.labelForField(self.scene_id_combo)
        self._scene_buttons = scene_buttons
        self.osm_panel = OsmScenePanel(scene_group)
        self.osm_panel.hide()
        self.osm_panel.view_changed.connect(self._osm_view_changed)
        self.osm_panel.map_entered.connect(self._enter_osm_map)
        self.osm_panel.map_left.connect(self._leave_osm_map)
        self.osm_panel.build_started.connect(self._osm_build_started)
        self.osm_panel.scene_ready.connect(self._install_osm_scene)
        self._osm_build_context = None
        self._osm_fit_pending = False
        self._osm_scroll = QScrollArea(scene_group)
        self._osm_scroll.setWidgetResizable(True)
        self._osm_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._osm_scroll.setFrameShape(QScrollArea.NoFrame)
        self._osm_scroll.setWidget(self.osm_panel)
        self._osm_scroll.setMinimumHeight(330)
        self._osm_scroll.setMaximumHeight(440)
        self._osm_scroll.hide()
        scene_form.addRow(self._osm_scroll)
        self.environment_library_button = QPushButton("Environment library…", scene_group)
        self.environment_library_button.clicked.connect(lambda: self.builder_tools.environments())
        scene_form.addRow(self.environment_library_button)
        layout.addWidget(scene_group)
        self._populate_scene_ids()

        add_row = QHBoxLayout()
        self.add_actor_button = QPushButton("Add actor", panel)
        self.add_actor_menu = QMenu(self.add_actor_button)
        self.add_actor_button.setMenu(self.add_actor_menu)
        add_row.addWidget(self.add_actor_button)
        self._add_actor_buttons = [self.add_actor_button]
        for label, role in (
            ("+ TX", ActorRole.TX),
            ("+ RX", ActorRole.RX),
            ("+ Target", ActorRole.TARGET),
        ):
            action = self.add_actor_menu.addAction(label.removeprefix("+ "))
            action.triggered.connect(lambda _checked=False, value=role: self._add_actor(value))
        self._form_group_button = QPushButton("Form Group...")
        self._form_group_button.setToolTip(
            "Choose two or more actors that should share one motion."
        )
        self._form_group_button.clicked.connect(self._form_group)
        add_row.addWidget(self._form_group_button)
        layout.addLayout(add_row)
        tools_row = QHBoxLayout()
        self.street_actors_button = QPushButton("Street actors…")
        self.street_actors_button.clicked.connect(lambda: self.builder_tools.street_actors())
        self.street_actors_button.hide()
        self.add_actor_menu.addAction(
            "Batch placement on streets…", lambda: self.builder_tools.street_actors()
        )
        self.placement_checks_button = QPushButton("Check placement")
        self.placement_checks_button.clicked.connect(lambda: self.builder_tools.check_positions())
        tools_row.addWidget(self.placement_checks_button)
        layout.addLayout(tools_row)
        self.actor_tree = QTreeWidget(panel)
        self.actor_tree.setHeaderLabels(("Actor / Group", "Role", "Visible", "Locked", "Status"))
        self.actor_tree.setSelectionMode(QTreeWidget.SingleSelection)
        self.actor_tree.itemSelectionChanged.connect(self._tree_selection_changed)
        self.actor_tree.itemChanged.connect(self._tree_item_changed)
        layout.addWidget(self.actor_tree, 1)
        return panel

    def _build_preview_summary(self) -> QWidget:
        """Keep validation and framing controls outside the overflowing tool bar."""

        panel = QWidget(self)
        panel.setObjectName("authoringPreviewSummary")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)
        status_row = QHBoxLayout()
        self.preview_status_label = QLabel("Validating preview…", panel)
        self.preview_status_label.setObjectName("authoringPreviewStatus")
        self.preview_status_label.setWordWrap(True)
        status_row.addWidget(self.preview_status_label, 1)
        self.show_problems_button = QPushButton("Show Problems", panel)
        self.show_problems_button.setObjectName("authoringShowProblemsButton")
        self.show_problems_button.clicked.connect(self.show_problems)
        status_row.addWidget(self.show_problems_button)
        layout.addLayout(status_row)

        camera_row = QHBoxLayout()
        self.focus_selection_button = QPushButton("Focus Selection", panel)
        self.focus_selection_button.clicked.connect(self.focus_selection)
        camera_row.addWidget(self.focus_selection_button)
        self.focus_on_select = QCheckBox("Focus on selection", panel)
        self.focus_on_select.setToolTip(
            "Move the camera to look at the selected actor's current position."
        )
        camera_row.addWidget(self.focus_on_select)
        self.camera_controls_button = QPushButton("Camera…", panel)
        self.camera_controls_button.clicked.connect(self._show_camera_controls)
        camera_row.addWidget(self.camera_controls_button)
        self.fit_actors_button = QPushButton("Fit Actors / Paths", panel)
        self.fit_actors_button.setObjectName("authoringFitActorsButton")
        self.fit_actors_button.setToolTip(
            "Frame visible actor positions and paths, including altitude, without the base scene."
        )
        self.fit_actors_button.clicked.connect(self.fit_actors)
        camera_row.addWidget(self.fit_actors_button)
        self.fit_scene_button = QPushButton("Fit Scene / All", panel)
        self.fit_scene_button.setObjectName("authoringFitSceneButton")
        self.fit_scene_button.setToolTip(
            "Frame the base scene together with visible actors and paths."
        )
        self.fit_scene_button.clicked.connect(self.fit_all)
        camera_row.addWidget(self.fit_scene_button)
        self.map_overlay_check = QCheckBox("2D map overlay", panel)
        self.map_overlay_check.setChecked(True)
        self.map_overlay_check.setEnabled(False)
        self.map_overlay_check.setVisible(False)
        self.map_overlay_check.setToolTip(
            "Show the map saved with this OSM scene. It does not change simulation geometry."
        )
        self.map_overlay_check.toggled.connect(lambda _: self._refresh_viewport())
        camera_row.addWidget(self.map_overlay_check)
        camera_row.addStretch()
        layout.addLayout(camera_row)
        return panel

    def _build_viewport_toolbar(self) -> QToolBar:
        toolbar = QToolBar("Authoring tools", self)
        toolbar.setObjectName("authoringViewportToolbar")
        self._tool_actions: dict[AuthoringTool, QAction] = {}
        for tool, label in (
            (AuthoringTool.SELECT, "Select"),
            (AuthoringTool.MOVE, "Edit motion in 3D"),
        ):
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked=False, value=tool: self.set_tool(
                    value if checked else AuthoringTool.SELECT
                )
            )
            self._tool_actions[tool] = action
        self._tool_actions[AuthoringTool.SELECT].setChecked(True)
        self._tool_actions[AuthoringTool.MOVE].setToolTip(
            "Use model handles, path dragging, or XYZ arrows to move the motion. "
            "Rotation rings edit a fixed orientation."
        )
        self.tool_hint_label = QLabel("Select an actor", toolbar)
        toolbar.addWidget(self.tool_hint_label)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Camera", toolbar))
        self.camera_mode_combo = QComboBox(toolbar)
        self.camera_mode_combo.addItem("Orbit", "orbit")
        self.camera_mode_combo.addItem("Fly", "fly")
        self.camera_mode_combo.currentIndexChanged.connect(self._camera_mode_changed)
        toolbar.addWidget(self.camera_mode_combo)
        toolbar.addSeparator()
        self.work_plane_check = QCheckBox("Work plane", toolbar)
        self.work_plane_check.setChecked(True)
        self.work_plane_check.setToolTip(
            "Show the horizontal placement plane and allow empty-space placement."
        )
        toolbar.addWidget(self.work_plane_check)
        toolbar.addWidget(QLabel("Z", toolbar))
        self.work_plane_z_spin = QDoubleSpinBox(toolbar)
        self.work_plane_z_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.work_plane_z_spin.setDecimals(3)
        self.work_plane_z_spin.setSuffix(" m")
        toolbar.addWidget(self.work_plane_z_spin)
        self.grid_snap_check = QCheckBox("Grid snap", toolbar)
        toolbar.addWidget(self.grid_snap_check)
        self.grid_snap_spacing_spin = QDoubleSpinBox(toolbar)
        self.grid_snap_spacing_spin.setRange(0.001, 1_000_000.0)
        self.grid_snap_spacing_spin.setDecimals(3)
        self.grid_snap_spacing_spin.setValue(1.0)
        self.grid_snap_spacing_spin.setSuffix(" m")
        self.grid_snap_spacing_spin.setEnabled(False)
        toolbar.addWidget(self.grid_snap_spacing_spin)
        self.work_plane_check.toggled.connect(self._viewport_settings_changed)
        self.work_plane_z_spin.valueChanged.connect(self._viewport_settings_changed)
        self.grid_snap_check.toggled.connect(self._viewport_settings_changed)
        self.grid_snap_spacing_spin.valueChanged.connect(self._viewport_settings_changed)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Trajectory", toolbar))
        self.trajectory_visibility_combo = QComboBox(toolbar)
        for label, value in (
            ("Off", OverlayVisibility.OFF),
            ("Selected", OverlayVisibility.SELECTED),
            ("All", OverlayVisibility.ALL),
        ):
            self.trajectory_visibility_combo.addItem(label, value)
        self.trajectory_visibility_combo.setCurrentIndex(
            self.trajectory_visibility_combo.findData(OverlayVisibility.ALL)
        )
        self.trajectory_visibility_combo.setToolTip(
            "Show the evaluated route for none, the selected actor, or all actors."
        )
        self.trajectory_visibility_combo.currentIndexChanged.connect(
            self._viewport_settings_changed
        )
        toolbar.addWidget(self.trajectory_visibility_combo)
        toolbar.addWidget(QLabel("Frames", toolbar))
        self.frame_samples_visibility_combo = QComboBox(toolbar)
        for label, value in (
            ("Off", OverlayVisibility.OFF),
            ("Selected", OverlayVisibility.SELECTED),
            ("All", OverlayVisibility.ALL),
        ):
            self.frame_samples_visibility_combo.addItem(label, value)
        self.frame_samples_visibility_combo.setCurrentIndex(
            self.frame_samples_visibility_combo.findData(OverlayVisibility.OFF)
        )
        self.frame_samples_visibility_combo.setToolTip(
            "Show the exact generator frame samples. Selected samples appear automatically "
            "during playback."
        )
        self.frame_samples_visibility_combo.currentIndexChanged.connect(
            self._viewport_settings_changed
        )
        toolbar.addWidget(self.frame_samples_visibility_combo)
        toolbar.addWidget(QLabel("Controls", toolbar))
        self.control_rig_visibility_combo = QComboBox(toolbar)
        for label, value in (
            ("Off", OverlayVisibility.OFF),
            ("Selected", OverlayVisibility.SELECTED),
            ("All", OverlayVisibility.ALL),
        ):
            self.control_rig_visibility_combo.addItem(label, value)
        self.control_rig_visibility_combo.setCurrentIndex(
            self.control_rig_visibility_combo.findData(OverlayVisibility.SELECTED)
        )
        self.control_rig_visibility_combo.setToolTip(
            "Show model-specific control handles while Edit Motion is active."
        )
        self.control_rig_visibility_combo.currentIndexChanged.connect(
            self._viewport_settings_changed
        )
        toolbar.addWidget(self.control_rig_visibility_combo)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Axes", toolbar))
        self.orientation_axes_combo = QComboBox(toolbar)
        for label, value in (
            ("Off", OverlayVisibility.OFF),
            ("Selected", OverlayVisibility.SELECTED),
            ("All", OverlayVisibility.ALL),
        ):
            self.orientation_axes_combo.addItem(label, value)
        self.orientation_axes_combo.setCurrentIndex(
            self.orientation_axes_combo.findData(OverlayVisibility.SELECTED)
        )
        self.orientation_axes_combo.setToolTip(
            "Show prepared orientation axes for none, the selected actor, or all actors. "
            "Derived modes appear once generator preparation succeeds."
        )
        self.orientation_axes_combo.currentIndexChanged.connect(self._viewport_settings_changed)
        toolbar.addWidget(self.orientation_axes_combo)
        toolbar.addWidget(QLabel("Look-at rays", toolbar))
        self.look_at_rays_combo = QComboBox(toolbar)
        for label, value in (
            ("Off", OverlayVisibility.OFF),
            ("Selected", OverlayVisibility.SELECTED),
            ("All", OverlayVisibility.ALL),
        ):
            self.look_at_rays_combo.addItem(label, value)
        self.look_at_rays_combo.setCurrentIndex(
            self.look_at_rays_combo.findData(OverlayVisibility.SELECTED)
        )
        self.look_at_rays_combo.setToolTip(
            "Show look-at relationship rays for none, the selected actor, or all actors."
        )
        self.look_at_rays_combo.currentIndexChanged.connect(self._viewport_settings_changed)
        toolbar.addWidget(self.look_at_rays_combo)
        return toolbar

    def _build_waypoint_session_bar(self) -> QWidget:
        """Build the persistent command surface for modal waypoint drawing."""

        bar = QWidget(self)
        bar.setObjectName("waypointDrawingSessionBar")
        bar.setStyleSheet(
            "#waypointDrawingSessionBar {"
            " background: #28445c; border: 1px solid #5d9bc8; border-radius: 3px;"
            "}"
        )
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 4, 8, 4)
        self.waypoint_session_label = QLabel("Drawing waypoints", bar)
        self.waypoint_session_label.setObjectName("waypointDrawingSessionLabel")
        layout.addWidget(self.waypoint_session_label, 1)
        self.waypoint_remove_last_button = QPushButton("Remove last", bar)
        self.waypoint_remove_last_button.setObjectName("waypointDrawingRemoveLastButton")
        self.waypoint_finish_button = QPushButton("Finish", bar)
        self.waypoint_finish_button.setObjectName("waypointDrawingFinishButton")
        self.waypoint_cancel_button = QPushButton("Cancel", bar)
        self.waypoint_cancel_button.setObjectName("waypointDrawingCancelButton")
        self.waypoint_remove_last_button.clicked.connect(self._remove_last_waypoint)
        self.waypoint_finish_button.clicked.connect(self._finish_waypoint_session)
        self.waypoint_cancel_button.clicked.connect(self._cancel_waypoint_session_and_exit)
        layout.addWidget(self.waypoint_remove_last_button)
        layout.addWidget(self.waypoint_finish_button)
        layout.addWidget(self.waypoint_cancel_button)
        bar.setVisible(False)
        self.waypoint_session_bar = bar
        return bar

    def _build_inspector(self) -> QWidget:
        panel = QWidget(self)
        self.inspector_panel = panel
        panel.setObjectName("authoringInspector")
        panel.setMinimumWidth(270)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.addWidget(QLabel("Inspector"))

        self.pending_inspector_banner = QWidget(panel)
        self.pending_inspector_banner.setObjectName("pendingInspectorBanner")
        self.pending_inspector_banner.setAccessibleName("Pending inspector edits")
        self.pending_inspector_banner.setStyleSheet(
            "#pendingInspectorBanner {"
            " background: #243f50; border: 1px solid #38bfd1; border-radius: 3px;"
            "}"
        )
        pending_layout = QVBoxLayout(self.pending_inspector_banner)
        pending_layout.setContentsMargins(7, 5, 7, 5)
        self.pending_inspector_label = QLabel("", self.pending_inspector_banner)
        self.pending_inspector_label.setObjectName("pendingInspectorLabel")
        self.pending_inspector_label.setWordWrap(True)
        pending_layout.addWidget(self.pending_inspector_label)
        pending_actions = QHBoxLayout()
        self.pending_apply_button = QPushButton(
            "Apply changes",
            self.pending_inspector_banner,
        )
        self.pending_apply_button.setObjectName("pendingInspectorApplyButton")
        self.pending_apply_button.clicked.connect(self._apply_all_pending_inspector_edits)
        self.pending_reset_button = QPushButton(
            "Discard",
            self.pending_inspector_banner,
        )
        self.pending_reset_button.setObjectName("pendingInspectorResetButton")
        self.pending_reset_button.clicked.connect(self._reset_all_pending_inspector_edits)
        pending_actions.addWidget(self.pending_apply_button)
        pending_actions.addWidget(self.pending_reset_button)
        pending_layout.addLayout(pending_actions)
        self.pending_inspector_banner.setVisible(False)
        layout.addWidget(self.pending_inspector_banner)

        self.inspector_scroll = InspectorScrollArea(panel)
        self.inspector_scroll.setObjectName("authoringInspectorScroll")
        self.inspector_scroll.setAccessibleName("Inspector sections")
        self.inspector_scroll.setWidgetResizable(True)
        self.inspector_scroll.setFrameShape(QScrollArea.NoFrame)
        self.inspector_scroll.setMinimumHeight(120)
        sections = QWidget(self.inspector_scroll)
        sections.setObjectName("authoringInspectorSections")
        self.inspector_scroll.setWidget(sections)
        layout.addWidget(self.inspector_scroll, 1)
        # Preserve each form's minimum usable height. The scroll area absorbs
        # overflow while the pending Apply/Reset banner stays reachable above it.
        layout = QVBoxLayout(sections)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSizeConstraint(QLayout.SetMinimumSize)

        identity = QGroupBox("Identity", panel)
        identity_form = AlignedFormLayout(identity)
        self.name_edit = QLineEdit(identity)
        self.name_edit.editingFinished.connect(self._apply_name)
        self.role_label = QLabel("—", identity)
        self.role_label.setWordWrap(True)
        identity_form.addRow("Name", self.name_edit)
        identity_form.addRow("Role", self.role_label)
        self.motion_edit_button = QToolButton(identity)
        self.motion_edit_button.setDefaultAction(self._tool_actions[AuthoringTool.MOVE])
        self.motion_edit_button.setStyleSheet("QToolButton:disabled { color: palette(mid); }")
        identity_form.addRow("Motion", self.motion_edit_button)
        self.motion_edit_hint = QLabel(identity)
        self.motion_edit_hint.setWordWrap(True)
        identity_form.addRow(self.motion_edit_hint)
        layout.addWidget(identity)
        self.animation_notice = QLabel(sections)
        self.animation_notice.setWordWrap(True)
        self.animation_notice.hide()
        layout.addWidget(self.animation_notice)

        mobility = QGroupBox("Mobility", panel)
        self.mobility_group = mobility
        mobility_layout = QVBoxLayout(mobility)
        self.mobility_editor = MobilityEditor(mobility)
        self.mobility_editor.mobility_type_changed.connect(self._mobility_type_changed)
        self.mobility_editor.apply_requested.connect(self._apply_mobility)
        self.mobility_editor.draw_waypoints_requested.connect(self._start_waypoint_rebuild)
        mobility_layout.addWidget(self.mobility_editor)
        self.mobility_editor.apply_button.hide()
        self.street_route_button = QPushButton("Choose named streets…", mobility)
        self.street_route_button.clicked.connect(
            lambda: self.builder_tools.street_actors(edit_selected=True)
        )
        mobility_layout.addWidget(self.street_route_button)
        self.mobility_reset_button = QPushButton("Reset Mobility", mobility)
        self.mobility_reset_button.setObjectName("mobilityResetButton")
        self.mobility_reset_button.clicked.connect(self._reset_pending_mobility)
        self.mobility_reset_button.setVisible(False)
        mobility_layout.addWidget(self.mobility_reset_button)
        layout.addWidget(mobility)

        self.group_settings = QGroupBox("Group Random Jitter", panel)
        group_form = AlignedFormLayout(self.group_settings)
        self.group_deviation_enabled = QCheckBox("Enabled", self.group_settings)
        group_form.addRow("Seeded per-frame jitter", self.group_deviation_enabled)
        self.group_deviation_spins: list[QDoubleSpinBox] = []
        for label, object_name in (
            ("Maximum right jitter", "groupDeviationRightSpin"),
            ("Maximum forward jitter", "groupDeviationForwardSpin"),
            ("Maximum up jitter", "groupDeviationUpSpin"),
        ):
            spin = QDoubleSpinBox(self.group_settings)
            spin.setObjectName(object_name)
            spin.setRange(0.0, 1_000_000_000.0)
            spin.setDecimals(6)
            spin.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)
            spin.setSuffix(" m")
            group_form.addRow(label, spin)
            self.group_deviation_spins.append(spin)
        self.group_deviation_seed = QSpinBox(self.group_settings)
        self.group_deviation_seed.setRange(0, MAX_RANDOM_SEED)
        group_form.addRow("Seed", self.group_deviation_seed)
        self.group_settings_apply_button = QPushButton(
            "Apply Group Settings",
            self.group_settings,
        )
        self.group_settings_apply_button.setObjectName("groupSettingsApplyButton")
        self.group_settings_apply_button.clicked.connect(self._apply_group_settings)
        self.group_settings_reset_button = QPushButton(
            "Reset Group Settings",
            self.group_settings,
        )
        self.group_settings_reset_button.setObjectName("groupSettingsResetButton")
        self.group_settings_reset_button.clicked.connect(self._reset_pending_group_settings)
        self.group_settings_reset_button.setVisible(False)
        group_form.addRow(self.group_settings_apply_button)
        self.group_settings_apply_button.hide()
        group_form.addRow(self.group_settings_reset_button)
        layout.addWidget(self.group_settings)

        orientation = QGroupBox("Orientation", panel)
        self.orientation_group = orientation
        orientation_layout = QVBoxLayout(orientation)
        self.orientation_editor = OrientationEditor(orientation)
        self.orientation_editor.orientation_type_changed.connect(self._orientation_type_changed)
        self.orientation_editor.apply_requested.connect(self._apply_orientation)
        orientation_layout.addWidget(self.orientation_editor)
        self.orientation_editor.apply_button.hide()
        self.orientation_reset_button = QPushButton("Reset Orientation", orientation)
        self.orientation_reset_button.setObjectName("orientationResetButton")
        self.orientation_reset_button.clicked.connect(self._reset_pending_orientation)
        self.orientation_reset_button.setVisible(False)
        orientation_layout.addWidget(self.orientation_reset_button)
        layout.addWidget(orientation)

        self.target_group = QGroupBox("Target asset", panel)
        target_form = AlignedFormLayout(self.target_group)
        self.target_representation_editor = TargetRepresentationEditor(self.target_group)
        self.target_asset_combo = QComboBox(self.target_group)
        self.target_asset_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.target_asset_combo.setMinimumContentsLength(12)
        self._populate_target_asset_choices()
        self.target_material_combo = QComboBox(self.target_group)
        for material_type in available_target_material_types():
            self.target_material_combo.addItem(material_type, material_type)
        self.target_material_combo.setCurrentIndex(self.target_material_combo.findData("glass"))
        self.target_scale_spin = QDoubleSpinBox(self.target_group)
        self.target_scale_spin.setRange(0.001, 1_000.0)
        self.target_scale_spin.setValue(1.0)
        self.target_mesh_animation = QCheckBox("Enabled", self.target_group)
        self.target_mesh_animation.setChecked(True)
        self.target_apply_button = QPushButton("Apply Target", self.target_group)
        self.target_apply_button.setObjectName("targetApplyButton")
        self.target_apply_button.clicked.connect(self._apply_target)
        self.target_reset_button = QPushButton("Reset Target", self.target_group)
        self.target_reset_button.setObjectName("targetResetButton")
        self.target_reset_button.clicked.connect(self._reset_pending_target)
        self.target_reset_button.setVisible(False)
        target_form.addRow("Catalog asset", self.target_asset_combo)
        target_form.addRow(self.target_representation_editor)
        target_form.addRow("Material", self.target_material_combo)
        target_form.addRow("Uniform scale", self.target_scale_spin)
        target_form.addRow("Mesh animation", self.target_mesh_animation)
        target_form.addRow(self.target_apply_button)
        self.target_apply_button.hide()
        target_form.addRow(self.target_reset_button)
        self.target_group.setVisible(False)
        self.target_representation_editor.changed.connect(self._target_representation_changed)
        self.group_settings.setVisible(False)
        layout.insertWidget(layout.indexOf(self.mobility_group), self.target_group)

        self.inline_errors = QLabel("", panel)
        self.inline_errors.setWordWrap(True)
        self.inline_errors.setStyleSheet("color: #df5c5c;")
        layout.addWidget(self.inline_errors)
        layout.addStretch()
        self._connect_editor_draft_signals(self.mobility_editor, "mobility")
        self._connect_editor_draft_signals(self.orientation_editor, "orientation")
        self.group_deviation_enabled.toggled.connect(
            lambda _checked: self._mark_inspector_draft("group_settings")
        )
        for spin in self.group_deviation_spins:
            spin.valueChanged.connect(lambda _value: self._mark_inspector_draft("group_settings"))
        self.group_deviation_seed.valueChanged.connect(
            lambda _value: self._mark_inspector_draft("group_settings")
        )
        self.target_asset_combo.currentIndexChanged.connect(
            lambda _index: self._mark_inspector_draft("target")
        )
        self.target_material_combo.currentIndexChanged.connect(
            lambda _index: self._mark_inspector_draft("target")
        )
        self.target_scale_spin.valueChanged.connect(
            lambda _value: self._mark_inspector_draft("target")
        )
        self.target_mesh_animation.toggled.connect(
            lambda _checked: self._mark_inspector_draft("target")
        )
        for form in sections.findChildren(QFormLayout):
            form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        return panel

    def _build_drawer(self) -> QTabWidget:
        drawer = QTabWidget(self)
        drawer.setObjectName("authoringDrawer")
        drawer.setMinimumHeight(190)

        timeline_page = QWidget(drawer)
        timeline_form = AlignedFormLayout(timeline_page)
        self.timeline_slider = QSlider(Qt.Horizontal, timeline_page)
        self.timeline_slider.setRange(0, 29)
        self.timeline_slider.valueChanged.connect(self._timeline_scrubbed)
        playback_row = QWidget(timeline_page)
        playback_layout = QHBoxLayout(playback_row)
        playback_layout.setContentsMargins(0, 0, 0, 0)
        self.play_button = QPushButton("Play", playback_row)
        self.play_button.clicked.connect(self._toggle_playback)
        playback_layout.addWidget(self.play_button)
        self.steps_spin = QSpinBox(timeline_page)
        self.steps_spin.setRange(1, 1_000_000)
        self.duration_spin = QDoubleSpinBox(timeline_page)
        self.duration_spin.setRange(0.0, 1_000_000.0)
        self.duration_spin.setDecimals(9)
        self.sample_rate_check = QCheckBox("Derive duration from sample rate", timeline_page)
        self.sample_rate_check.setToolTip(
            "The scene clock uses N endpoint-inclusive samples: duration = (N - 1) / rate. "
            "The scene rate is separate from a sensing processor's PRF."
        )
        self.sample_rate_spin = QDoubleSpinBox(timeline_page)
        self.sample_rate_spin.setDecimals(12)
        self.sample_rate_spin.setRange(1e-12, 1e15)
        self.sample_rate_spin.setValue(1000.0)
        self.sample_rate_spin.setEnabled(False)
        self.quality_combo = QComboBox(timeline_page)
        for value in QualityPreset:
            if value is QualityPreset.CUSTOM:
                continue
            self.quality_combo.addItem(value.value, value)
        self.path_metrics_check = QCheckBox("Export path metrics", timeline_page)
        self.sampling_summary_label = QLabel("", timeline_page)
        self.sampling_summary_label.setObjectName("authoringSamplingSummary")
        self.sampling_summary_label.setWordWrap(True)
        self.preserved_settings_label = QLabel("", timeline_page)
        self.preserved_settings_label.setObjectName("authoringPreservedSettings")
        self.preserved_settings_label.setWordWrap(True)
        self.preserved_settings_label.setStyleSheet("color: #d6a34a;")
        self.preserved_settings_label.setVisible(False)
        self.apply_timeline_button = QPushButton("Apply Timeline", timeline_page)
        self.apply_timeline_button.clicked.connect(self._apply_timeline)
        self.reset_timeline_button = QPushButton("Reset", timeline_page)
        self.reset_timeline_button.clicked.connect(self._reset_pending_timeline)
        self.timeline_pending_label = QLabel("", timeline_page)
        self.timeline_pending_label.setObjectName("authoringTimelinePending")
        self.timeline_pending_label.setWordWrap(True)
        self.timeline_pending_label.setStyleSheet("color: #d6a34a;")
        timeline_actions = QWidget(timeline_page)
        timeline_actions_layout = QHBoxLayout(timeline_actions)
        timeline_actions_layout.setContentsMargins(0, 0, 0, 0)
        timeline_actions_layout.addWidget(self.apply_timeline_button)
        timeline_actions_layout.addWidget(self.reset_timeline_button)
        timeline_actions_layout.addStretch()
        timeline_form.addRow("Steps", self.steps_spin)
        timeline_form.addRow(self.sample_rate_check)
        timeline_form.addRow("Sample rate (Hz)", self.sample_rate_spin)
        timeline_form.addRow("Duration (s)", self.duration_spin)
        timeline_form.addRow("Quality", self.quality_combo)
        timeline_form.addRow(self.path_metrics_check)
        timeline_form.addRow("Sampling", self.sampling_summary_label)
        timeline_form.addRow(self.preserved_settings_label)
        timeline_form.addRow(self.timeline_pending_label)
        timeline_form.addRow(timeline_actions)
        self.steps_spin.valueChanged.connect(self._mark_timeline_draft)
        self.duration_spin.valueChanged.connect(self._mark_timeline_draft)
        self.sample_rate_check.toggled.connect(self._mark_timeline_draft)
        self.sample_rate_spin.valueChanged.connect(self._mark_timeline_draft)
        self.quality_combo.currentIndexChanged.connect(self._mark_timeline_draft)
        self.path_metrics_check.toggled.connect(self._mark_timeline_draft)
        self._update_timeline_pending_ui()
        drawer.addTab(timeline_page, "Timeline")

        self.problems_tree = QTreeWidget(drawer)
        self.problems_tree.setHeaderLabels(("Severity", "Path", "Problem"))
        drawer.addTab(self.problems_tree, "Problems")
        self.yaml_preview = QPlainTextEdit(drawer)
        self.yaml_preview.setReadOnly(True)
        drawer.addTab(self.yaml_preview, "YAML Preview")
        self.generation_log = QPlainTextEdit(drawer)
        self.generation_log.setReadOnly(True)
        drawer.addTab(self.generation_log, "Generation Log")

        generation_page = QWidget(drawer)
        generation_layout = QVBoxLayout(generation_page)
        self.validation_status = QLabel(
            "Validation pending for the current draft.",
            generation_page,
        )
        self.validation_status.setObjectName("authoringValidationStatus")
        self.validation_status.setWordWrap(True)
        self.generation_status = QLabel("No generation has been launched.", generation_page)
        self.generation_progress = QProgressBar(generation_page)
        self.generation_progress.setRange(0, 100)
        self.generation_progress.setValue(0)
        generation_layout.addWidget(self.validation_status)
        generation_layout.addWidget(self.generation_status)
        generation_layout.addWidget(self.generation_progress)
        generation_buttons = QHBoxLayout()
        self.validate_button = QPushButton("Validate", generation_page)
        self.validate_button.setObjectName("authoringValidateButton")
        self.validate_button.clicked.connect(self.validate_current_draft)
        self.save_button = QPushButton("Save", generation_page)
        self.save_button.clicked.connect(self.save_requested.emit)
        self.save_as_button = QPushButton("Save As...", generation_page)
        self.save_as_button.clicked.connect(self.save_as_requested.emit)
        self.generate_button = QPushButton("Generate", generation_page)
        self.generate_button.clicked.connect(self.generate_requested.emit)
        self.cancel_generation_button = QPushButton("Cancel", generation_page)
        self.cancel_generation_button.setEnabled(False)
        self.cancel_generation_button.clicked.connect(self.cancel_generation_requested.emit)
        self.preview_result_button = QPushButton("Preview Generated Result", generation_page)
        self.preview_result_button.setEnabled(False)
        self.preview_result_button.clicked.connect(
            lambda: self.preview_generated_requested.emit(None)
        )
        for button in (
            self.validate_button,
            self.save_button,
            self.save_as_button,
            self.generate_button,
            self.cancel_generation_button,
            self.preview_result_button,
        ):
            generation_buttons.addWidget(button)
        generation_buttons.addStretch()
        generation_layout.addLayout(generation_buttons)
        drawer.addTab(generation_page, "Generation")
        from .sensing_inspector import SensingInspector

        self.sensing_inspector = SensingInspector(drawer)
        drawer.addTab(self.sensing_inspector, "Sensing (read-only)")
        from .builder_tools import BuilderTools

        self.builder_tools = BuilderTools(self)
        self.sensing_inspector.show_guides.toggled.connect(
            self.builder_tools.show_sensing.setChecked
        )
        self.builder_tools.show_sensing.toggled.connect(
            self.sensing_inspector.show_guides.setChecked
        )
        drawer.addTab(self.builder_tools, "Scene tools")
        port = getattr(self.viewport, "port", None)
        if callable(getattr(port, "preview_bounds", None)):
            from .camera import BuilderCamera

            self.camera_controls = BuilderCamera(self, port)
            self.camera_scroll = QScrollArea(drawer)
            self.camera_scroll.setWidgetResizable(True)
            self.camera_scroll.setWidget(self.camera_controls)
            drawer.addTab(self.camera_scroll, "Camera")
        self.camera_controls_button.setEnabled(self.camera_controls is not None)
        return drawer
