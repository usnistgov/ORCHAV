"""Startup window and panel composition helpers.

This module contains Qt-shell work that must happen around deferred startup:
identity/theme application, the lightweight loading placeholder, and final
control-panel construction once services and a renderer object exist.
"""

from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QProgressBar,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from shared.logging import get_logger

from ..renderers.protocol import renderer_capabilities
from .app_identity import apply_application_identity, apply_window_identity
from .theme import get_theme_manager
from .ui_setup import build_control_panel
from .viewport_workspace import ViewportState, VisualizationWorkspace
from .window_manager import (
    GAP_PX,
    apply_qt_layout,
    compute_layout,
    qt_min_width_for_profile,
)

logger = get_logger(__name__)


def apply_visualizer_shell(
    viz: Any,
    *,
    layout_profile: str = "auto",
    viewport_mode: str = "embedded",
) -> None:
    """Apply application identity, theme, and initial window geometry."""
    app = QApplication.instance()
    if app is not None:
        apply_application_identity(app)
        get_theme_manager().install(app)

    apply_window_identity(viz)
    minimum_width = qt_min_width_for_profile(layout_profile)
    if viewport_mode == "embedded":
        minimum_width = max(900, minimum_width)
    viz.setMinimumSize(minimum_width, 500)

    screen = QApplication.primaryScreen()
    if screen is not None:
        geom = screen.availableGeometry()
        device_pixel_ratio = 1.0
        try:
            device_pixel_ratio = float(screen.devicePixelRatio())
        except (AttributeError, TypeError, ValueError):
            logger.debug("Could not read screen device pixel ratio", exc_info=True)
        viz._window_layout = compute_layout(
            geom.width(),
            geom.height(),
            geom.x(),
            geom.y(),
            layout_profile=layout_profile,
            device_pixel_ratio=device_pixel_ratio,
        )
        if viewport_mode == "embedded":
            layout = viz._window_layout
            viz.setGeometry(
                layout.qt_x,
                layout.qt_y,
                layout.qt_width + GAP_PX + layout.renderer_logical_width,
                layout.qt_height,
            )
        else:
            apply_qt_layout(viz, viz._window_layout)
    else:
        viz._window_layout = None
        viz.resize(520, 800)


def install_loading_placeholder(viz: Any) -> None:
    """Install the lightweight loading placeholder shown before deferred init."""
    viz._workspace_stack = QStackedWidget(viz)
    viz._workspace_stack.setObjectName("applicationWorkspaceStack")
    viz._loading_widget = QWidget(viz._workspace_stack)
    loading_layout = QVBoxLayout(viz._loading_widget)
    loading_layout.setContentsMargins(20, 20, 20, 20)
    viz._loading_label = QLabel("Loading...")
    viz._loading_label.setAlignment(Qt.AlignCenter)
    viz._loading_label.setStyleSheet("font-size: 16px;")
    loading_layout.addWidget(viz._loading_label)
    viz._loading_progress = QProgressBar()
    viz._loading_progress.setRange(0, 0)
    loading_layout.addWidget(viz._loading_progress)
    viz._workspace_stack.addWidget(viz._loading_widget)
    viz._workspace_stack.setCurrentWidget(viz._loading_widget)
    viz.setCentralWidget(viz._workspace_stack)


def make_loading_updater(viz: Any):
    """Return a throttled loading-label updater for deferred initialization."""
    last_process_events_time = [0.0]

    def update_loading(message: str) -> None:
        """Update placeholder text while throttling nested Qt event processing."""
        if bool(getattr(viz, "_shutdown_started", False)):
            raise RuntimeError("Visualizer initialization cancelled during shutdown")
        viz._loading_label.setText(message)
        now = time.monotonic()
        if now - last_process_events_time[0] > 0.05:
            QApplication.processEvents()
            last_process_events_time[0] = now
            if bool(getattr(viz, "_shutdown_started", False)):
                raise RuntimeError("Visualizer initialization cancelled during shutdown")

    return update_loading


def build_main_panels(viz: Any, *, metrics_available: bool) -> None:
    """Build the central control panel and apply renderer-dependent UI state.

    Panel construction is renderer-neutral, then capability flags reveal or
    hide controls whose behavior depends on the selected backend.
    """
    viz.ui_controller.setup_menus(viz, metrics_available=metrics_available)

    viz.ui_manager, viz.ctrl_panel = build_control_panel(viz, viz.total_animation_steps)
    embedded = getattr(viz, "_viewport_mode", "embedded") == "embedded"
    viz._visualization_workspace = VisualizationWorkspace(
        viz.ctrl_panel,
        embedded=embedded,
        parent=viz._workspace_stack,
    )
    viz._viewport_host = viz._visualization_workspace.viewport_host
    viz._viewport_host.open_requested.connect(viz.ui_controller.open_scenario_dialog)
    viz._viewport_host.retry_requested.connect(viz.retry_last_scenario)
    viz._viewport_host.logical_size_changed.connect(viz._on_viewport_resized)
    viz._viewport_host.screen_changed.connect(viz._on_viewport_screen_changed)
    if embedded:
        viz._viewport_host.set_state(ViewportState.EMPTY)
    viz._workspace_stack.addWidget(viz._visualization_workspace)
    viz._workspace_stack.setCurrentWidget(viz._visualization_workspace)
    loading_widget = getattr(viz, "_loading_widget", None)
    if loading_widget is not None:
        viz._workspace_stack.removeWidget(loading_widget)
        loading_widget.deleteLater()

    # Show fly mode checkbox only when backend supports fly controls.
    capabilities = renderer_capabilities(viz.renderer)
    if capabilities.fly_mode and getattr(viz, "fly_mode_cb", None):
        viz.fly_mode_cb.setVisible(True)
        viz.fly_mode_cb.setChecked(bool(getattr(viz.app_state, "fly_mode", False)))
    if capabilities.open3d_settings_panel:
        follow_note = (
            "Auto-follow selected entity (camera looks at entity).\n"
            "Open3D renderer limitation: wheel zoom during Follow is not persistent and "
            "may reset on playback updates."
        )
        if getattr(viz, "follow_mode_rb", None):
            viz.follow_mode_rb.setToolTip(follow_note)
        if getattr(viz, "target_focus_dropdown", None):
            viz.target_focus_dropdown.setToolTip(
                "Select entity to track in Follow/POV mode.\n"
                "On the Open3D renderer, Follow keeps tracking correctly but zoom changes "
                "can be overwritten while playback is running."
            )
    if capabilities.camera_minimap and getattr(viz, "camera_minimap_cb", None):
        viz.camera_minimap_cb.setVisible(True)
        viz.camera_minimap_cb.setChecked(bool(getattr(viz.app_state, "show_camera_minimap", False)))
        try:
            viz.renderer.set_camera_minimap_visible(
                bool(getattr(viz.app_state, "show_camera_minimap", False))
            )
        except (AttributeError, RuntimeError, ValueError):
            logger.debug("Could not apply initial minimap state", exc_info=True)

    supports_trajectories = capabilities.trajectories
    nodes_panel = viz.ui_manager.panels.get("nodes")
    if nodes_panel is not None and hasattr(nodes_panel, "widgets"):
        trajectory_widget_keys = (
            "tx_trajectory_cb",
            "rx_trajectory_cb",
            "target_trajectory_cb",
            "trajectory_status_label",
            "trajectory_line_width_spin",
            "trajectory_point_size_spin",
            "trajectory_colorbar_container",
            "trajectory_color_node_color_rb",
            "trajectory_color_speed_rb",
            "trajectory_color_altitude_rb",
            "trajectory_color_time_rb",
            "trajectory_color_angular_speed_rb",
        )
        for key in trajectory_widget_keys:
            widget = nodes_panel.widgets.get(key)
            if widget is not None:
                widget.setVisible(supports_trajectories)
        status_label = nodes_panel.widgets.get("trajectory_status_label")
        if status_label is not None and not supports_trajectories:
            status_label.setText("Trajectories unsupported by current renderer")
        refresh_preview = getattr(nodes_panel, "refresh_live_preview_state", None)
        if callable(refresh_preview):
            refresh_preview()

    export_panel = viz.ui_manager.panels.get("export")
    supports_screenshot_export = capabilities.screenshot_export
    if export_panel is not None and hasattr(export_panel, "set_screenshot_export_enabled"):
        export_panel.set_screenshot_export_enabled(supports_screenshot_export)

    if "mpc" in viz.ui_manager.panels:
        viz.ui_manager.panels["mpc"].refresh_preset_list()

    viz._compact_mode = False
