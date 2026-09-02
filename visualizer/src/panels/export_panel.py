"""Export entry points for renderer screenshots and animation captures.

The panel owns only the buttons and default screenshot filename policy.
Renderer capability checks and the video-export dialog perform the actual
capture work.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from shared.logging import get_logger

from ..renderers.protocol import renderer_capabilities
from .base import BasePanel

logger = get_logger("orchav.export_panel")

_EXPORT_BTN_STYLE = """
    QPushButton {
        background-color: #27ae60;
        color: #ffffff;
        border: 1px solid #229954;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 11px;
    }
    QPushButton:hover { background-color: #2ecc71; }
    QPushButton:pressed { background-color: #229954; }
"""


class ExportPanel(BasePanel):
    """Create screenshot and video/GIF export buttons."""

    def __init__(self, parent_widget, total_steps: int = 60) -> None:
        """Store timeline length for the video export dialog."""
        super().__init__(parent_widget)
        self.total_steps = total_steps

    def create_panel(self) -> QGroupBox:
        """Build export controls and connect them to local dialog handlers."""
        group = self.create_group_box("Export")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        options_row = QHBoxLayout()
        options_row.addWidget(QLabel("Output scale:"))
        self.widgets["screenshot_resolution_spin"] = QDoubleSpinBox()
        self.widgets["screenshot_resolution_spin"].setRange(0.25, 4.0)
        self.widgets["screenshot_resolution_spin"].setSingleStep(0.25)
        self.widgets["screenshot_resolution_spin"].setValue(1.0)
        self.widgets["screenshot_resolution_spin"].setSuffix("×")
        self.widgets["screenshot_resolution_spin"].setToolTip(
            "Post-capture bilinear resize (1× keeps native renderer pixels; "
            "larger values do not add scene detail)"
        )
        options_row.addWidget(self.widgets["screenshot_resolution_spin"])

        self.widgets["include_hud_cb"] = QCheckBox("Include viewport HUD")
        self.widgets["include_hud_cb"].setChecked(False)
        self.widgets["include_hud_cb"].setToolTip(
            "Composite visible status, legend, and filter overlays into exports"
        )
        self.widgets["include_hud_cb"].setVisible(
            renderer_capabilities(getattr(self.parent, "renderer", None)).viewport_hud
        )
        options_row.addWidget(self.widgets["include_hud_cb"])
        options_row.addStretch()
        layout.addLayout(options_row)

        actions_row = QHBoxLayout()
        self.widgets["screenshot_btn"] = QPushButton("Screenshot")
        self.widgets["screenshot_btn"].setToolTip("Save current view as PNG image")
        self.widgets["screenshot_btn"].setStyleSheet(_EXPORT_BTN_STYLE)
        self.widgets["screenshot_btn"].clicked.connect(self._on_screenshot_clicked)
        actions_row.addWidget(self.widgets["screenshot_btn"])

        self.widgets["export_video_btn"] = QPushButton("Video / GIF")
        self.widgets["export_video_btn"].setToolTip("Export animation as MP4 video or GIF")
        self.widgets["export_video_btn"].setStyleSheet(_EXPORT_BTN_STYLE)
        self.widgets["export_video_btn"].clicked.connect(self._on_export_video_clicked)
        actions_row.addWidget(self.widgets["export_video_btn"])

        actions_row.addStretch()
        layout.addLayout(actions_row)
        return group

    def update_total_steps(self, new_total_steps: int) -> None:
        """Mirror scenario frame-count changes into future video dialogs."""
        self.total_steps = new_total_steps

    def set_screenshot_export_enabled(self, enabled: bool) -> None:
        """Enable/disable screenshot and video export controls."""
        btn = self.widgets.get("screenshot_btn")
        if btn is not None:
            btn.setEnabled(bool(enabled))
        video_btn = self.widgets.get("export_video_btn")
        if video_btn is not None:
            video_btn.setEnabled(bool(enabled))

    # Screenshot

    def _build_screenshot_default_name(self) -> str:
        """Build a default screenshot filename from scenario name and frame step."""
        parts: list[str] = []
        viz = self.parent
        if viz is not None:
            scenario_name = getattr(viz, "scenario_name", None)
            if not scenario_name:
                app_state = getattr(viz, "app_state", None)
                scenario_name = getattr(app_state, "scenario_name", None)
            if scenario_name:
                parts.append(scenario_name)
            app_state = getattr(viz, "app_state", None)
            step = getattr(app_state, "step", None)
            if step is not None:
                parts.append(f"frame_{step:04d}")
        if not parts:
            return "screenshot.png"
        return "_".join(parts) + ".png"

    def _on_screenshot_clicked(self) -> None:
        """Handle screenshot export button click."""
        if not hasattr(self.parent, "renderer") or self.parent.renderer is None:
            return
        renderer = self.parent.renderer
        if not renderer_capabilities(renderer).screenshot_export:
            return
        if not hasattr(renderer, "export_screenshot"):
            return

        default_name = self._build_screenshot_default_name()
        file_path, _ = QFileDialog.getSaveFileName(
            self.parent,
            "Export Screenshot",
            default_name,
            "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg)",
        )
        if file_path:
            scale_widget = self.widgets.get("screenshot_resolution_spin")
            hud_widget = self.widgets.get("include_hud_cb")
            resolution_scale = float(scale_widget.value()) if scale_widget is not None else 1.0
            include_hud = bool(hud_widget.isChecked()) if hud_widget is not None else False
            success = renderer.export_screenshot(
                file_path,
                resolution_scale=resolution_scale,
                include_hud=include_hud,
            )
            if hasattr(self.parent, "_set_status_message"):
                if success:
                    self.parent._set_status_message(f"Screenshot saved to {file_path}", 5000)
                else:
                    self.parent._set_status_message(
                        f"Failed to export screenshot to {file_path}", 5000
                    )

    # Video / GIF

    def _on_export_video_clicked(self) -> None:
        """Handle export video button click."""
        from .video_export_dialog import VideoExportDialog

        hud_widget = self.widgets.get("include_hud_cb")
        dialog = VideoExportDialog(
            self.parent,
            self.total_steps,
            include_hud=bool(hud_widget.isChecked()) if hud_widget is not None else False,
        )
        dialog.exec()
