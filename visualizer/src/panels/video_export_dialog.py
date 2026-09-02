"""Video/GIF export dialog for animation frame ranges.

The dialog gathers user-visible export settings and delegates actual rendering
to the parent visualizer's ``export_video`` method. Frame numbers are visualizer
display indices, while export stride follows the current animation stride so
preview/playback and exported clips use the same sparse-step policy.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from shared.logging import get_logger

from ..renderers.protocol import renderer_capabilities
from .ui_theme import configure_label

logger = get_logger("orchav.video_export_dialog")


class VideoExportDialog(QDialog):
    """Collect export settings and call the parent visualizer export API."""

    def __init__(self, parent, total_frames: int, *, include_hud: bool = False):
        """Initialize export settings for a visualizer with a fixed frame count."""
        super().__init__(parent)
        self.parent_viz = parent
        self.total_frames = total_frames
        self.setWindowTitle("Export Video/GIF")
        self.setModal(True)
        self.resize(500, 350)
        self._initial_include_hud = bool(include_hud)

        self._setup_ui()

    def _setup_ui(self):
        """Build controls for output path, format, frame range, FPS, and scale."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Output file selection
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Output File:"))
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Select output file path...")
        file_layout.addWidget(self.file_input, 1)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(self.browse_btn)
        layout.addLayout(file_layout)

        # Format selection
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["MP4 Video", "GIF Animation"])
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        layout.addLayout(format_layout)

        # Frame range
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Frame Range:"))

        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setMinimum(0)
        self.start_frame_spin.setMaximum(self.total_frames - 1)
        self.start_frame_spin.setValue(0)
        self.start_frame_spin.setPrefix("Start: ")
        range_layout.addWidget(self.start_frame_spin)

        range_layout.addWidget(QLabel("to"))

        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setMinimum(0)
        self.end_frame_spin.setMaximum(self.total_frames - 1)
        self.end_frame_spin.setValue(self.total_frames - 1)
        self.end_frame_spin.setPrefix("End: ")
        range_layout.addWidget(self.end_frame_spin)

        range_layout.addStretch()
        layout.addLayout(range_layout)

        # FPS selection
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Frame Rate (FPS):"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setMinimum(1)
        self.fps_spin.setMaximum(60)
        self.fps_spin.setValue(30)
        self.fps_spin.setToolTip("Frames per second (higher = smoother but larger file)")
        fps_layout.addWidget(self.fps_spin)
        fps_layout.addStretch()
        layout.addLayout(fps_layout)

        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel("Output Scale:"))
        self.resolution_spin = QDoubleSpinBox()
        self.resolution_spin.setMinimum(0.5)
        self.resolution_spin.setMaximum(2.0)
        self.resolution_spin.setSingleStep(0.1)
        self.resolution_spin.setValue(1.0)
        self.resolution_spin.setToolTip(
            "Post-capture bilinear resize (1.0 keeps native renderer pixels; "
            "larger values do not add scene detail)"
        )
        resolution_layout.addWidget(self.resolution_spin)
        resolution_layout.addStretch()
        layout.addLayout(resolution_layout)

        self.include_hud_cb = QCheckBox("Include viewport HUD")
        self.include_hud_cb.setChecked(self._initial_include_hud)
        self.include_hud_cb.setToolTip(
            "Composite visible status, legend, and filter overlays into every frame"
        )
        self.include_hud_cb.setVisible(
            renderer_capabilities(getattr(self.parent_viz, "renderer", None)).viewport_hud
        )
        layout.addWidget(self.include_hud_cb)

        # Info label
        self.info_label = QLabel("")
        configure_label(self.info_label, role="muted", font_size=10, italic=True)
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.export_btn = QPushButton("Export")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:pressed {
                background-color: #229954;
            }
        """)
        self.export_btn.clicked.connect(self._export)
        button_layout.addWidget(self.export_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #b2bec3;
            }
        """)
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

        self.start_frame_spin.valueChanged.connect(self._update_info)
        self.end_frame_spin.valueChanged.connect(self._update_info)
        self.fps_spin.valueChanged.connect(self._update_info)

        self._update_info()

    def _resolve_export_stride(self) -> int:
        """Return the active animation stride, falling back to no skipping."""
        viz = self.parent_viz
        if viz is None:
            return 1
        controller = getattr(viz, "animation_controller", None)
        if controller is None:
            return 1
        try:
            return max(1, int(controller.get_current_stride()))
        except (TypeError, ValueError):
            return 1

    def _browse_file(self):
        """Open a save dialog and choose the extension matching the format."""
        current_format = self.format_combo.currentText()
        if "MP4" in current_format:
            file_filter = "MP4 Video (*.mp4)"
            default_ext = ".mp4"
        else:
            file_filter = "GIF Animation (*.gif)"
            default_ext = ".gif"

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Select Output File",
            f"animation{default_ext}",
            file_filter,
        )

        if filename:
            self.file_input.setText(filename)
            self._update_info()

    def _on_format_changed(self):
        """Update the selected file suffix when MP4/GIF format changes."""
        current_path = self.file_input.text()
        if current_path:
            path = Path(current_path)
            current_format = self.format_combo.currentText()
            if "MP4" in current_format:
                new_path = path.with_suffix(".mp4")
            else:
                new_path = path.with_suffix(".gif")
            self.file_input.setText(str(new_path))

        self._update_info()

    def _update_info(self):
        """Refresh summary text from frame range, stride, and FPS."""
        start = self.start_frame_spin.value()
        end = self.end_frame_spin.value()
        fps = self.fps_spin.value()
        stride = self._resolve_export_stride()
        total = len(range(start, end + 1, stride)) if start <= end else 0

        duration = total / fps if fps > 0 else 0.0
        stride_text = f" with stride x{stride}" if stride > 1 else ""
        self.info_label.setText(
            f"Will export {total} frames at {fps} FPS{stride_text} "
            f"(about {duration:.1f} seconds)"
        )

    def _export(self):
        """Validate settings and delegate export to the parent visualizer."""
        output_path = self.file_input.text()
        if not output_path:
            QMessageBox.warning(self, "No Output File", "Please select an output file path.")
            return

        start_frame = self.start_frame_spin.value()
        end_frame = self.end_frame_spin.value()
        fps = self.fps_spin.value()
        resolution_scale = self.resolution_spin.value()
        stride = self._resolve_export_stride()

        if start_frame > end_frame:
            QMessageBox.warning(
                self, "Invalid Range", "Start frame must be less than or equal to end frame."
            )
            return

        logger.info(f"Exporting video: {output_path} (frames {start_frame}-{end_frame}, {fps} FPS)")

        success = self.parent_viz.export_video(
            output_path=output_path,
            fps=fps,
            start_frame=start_frame,
            end_frame=end_frame,
            stride=stride,
            resolution_scale=resolution_scale,
            include_hud=self.include_hud_cb.isChecked(),
        )

        if success:
            QMessageBox.information(
                self,
                "Export Complete",
                f"Video successfully exported to:\n{output_path}",
            )
            self.accept()
        else:
            error_text = (
                getattr(self.parent_viz, "last_video_export_error", "")
                or "Video export failed. Check the log for details."
            )
            QMessageBox.critical(
                self,
                "Export Failed",
                error_text,
            )
