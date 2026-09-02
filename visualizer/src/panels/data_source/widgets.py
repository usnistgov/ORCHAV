"""Reusable widgets for live data-source diagnostics.

These helpers render frame-source diagnostics without owning frame loading or
streaming policy. They read frame-source/provider payloads supplied by the
parent panel and keep their own display state small.
"""

from __future__ import annotations

from typing import Dict

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from shared.frames.types import StandardMPCFrame
from shared.logging import get_logger

logger = get_logger("orchav.data_source_panel.widgets")


class FrameComparisonDialog(QDialog):
    """Compare two loaded frames using lightweight MPC and node counts."""

    def __init__(self, parent):
        """Create frame selectors and side-by-side summary panes."""
        super().__init__(parent)
        self.setWindowTitle("Frame Comparison")
        self.setModal(True)
        self.resize(1000, 700)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        selection_layout = QHBoxLayout()
        selection_layout.addWidget(QLabel("Frame 1:"))
        self.frame1_spinbox = QSpinBox()
        self.frame1_spinbox.setRange(0, 1000)
        self.frame1_spinbox.setValue(0)
        selection_layout.addWidget(self.frame1_spinbox)

        selection_layout.addWidget(QLabel("Frame 2:"))
        self.frame2_spinbox = QSpinBox()
        self.frame2_spinbox.setRange(0, 1000)
        self.frame2_spinbox.setValue(1)
        selection_layout.addWidget(self.frame2_spinbox)

        load_btn = QPushButton("Load Frames")
        load_btn.clicked.connect(self._load_frames)
        selection_layout.addWidget(load_btn)
        selection_layout.addStretch()

        layout.addLayout(selection_layout)

        splitter = QSplitter(Qt.Horizontal)

        frame1_widget = QWidget()
        frame1_layout = QVBoxLayout(frame1_widget)
        frame1_layout.addWidget(QLabel("Frame 1"))
        self.frame1_info = QTextEdit()
        self.frame1_info.setReadOnly(True)
        self.frame1_info.setMaximumHeight(200)
        frame1_layout.addWidget(self.frame1_info)
        splitter.addWidget(frame1_widget)

        frame2_widget = QWidget()
        frame2_layout = QVBoxLayout(frame2_widget)
        frame2_layout.addWidget(QLabel("Frame 2"))
        self.frame2_info = QTextEdit()
        self.frame2_info.setReadOnly(True)
        self.frame2_info.setMaximumHeight(200)
        frame2_layout.addWidget(self.frame2_info)
        splitter.addWidget(frame2_widget)

        layout.addWidget(splitter)

        diff_label = QLabel("Differences:")
        diff_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(diff_label)

        self.diff_info = QTextEdit()
        self.diff_info.setReadOnly(True)
        self.diff_info.setMaximumHeight(150)
        layout.addWidget(self.diff_info)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.frame1_data: StandardMPCFrame | None = None
        self.frame2_data: StandardMPCFrame | None = None

    def _load_frames(self):
        """Load selected frames through the parent frame source and summarize them."""
        try:
            frame1_idx = self.frame1_spinbox.value()
            frame2_idx = self.frame2_spinbox.value()

            parent = self.parent()
            if not parent or not hasattr(parent, "frame_source") or parent.frame_source is None:
                QMessageBox.warning(self, "Error", "No frame source available")
                return

            frame_source = parent.frame_source
            if not hasattr(frame_source, "load_frame"):
                QMessageBox.warning(self, "Error", "Frame source does not support loading frames")
                return

            try:
                frame1 = frame_source.load_frame(frame1_idx)
                frame2 = frame_source.load_frame(frame2_idx)
            except (OSError, KeyError, ValueError) as e:
                logger.error(f"Error loading frames {frame1_idx} and {frame2_idx}: {e}")
                QMessageBox.warning(self, "Error", f"Failed to load frames: {e}")
                return

            if frame1 is None or frame2 is None:
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Failed to load one or both frames (Frame 1: {frame1_idx}, Frame 2: {frame2_idx})",
                )
                return

            self.frame1_data = frame1
            self.frame2_data = frame2

            self._display_frame_info(frame1, self.frame1_info, frame1_idx)
            self._display_frame_info(frame2, self.frame2_info, frame2_idx)

            self._display_differences(frame1, frame2)

        except (OSError, KeyError, ValueError, RuntimeError) as e:
            logger.error(f"Error loading frames for comparison: {e}", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to load frames: {e}")

    def _display_frame_info(
        self,
        frame: StandardMPCFrame,
        text_widget: QTextEdit,
        frame_idx: int,
    ) -> None:
        """Display canonical path and device counts for one frame."""
        try:
            info = (
                f"Frame {frame_idx}\n"
                f"MPC Count: {int(frame.num_paths or 0)}\n"
                f"TX Count: {frame.num_tx}\n"
                f"RX Count: {frame.num_rx}\n"
            )

            text_widget.setPlainText(info)
        except (KeyError, AttributeError, TypeError) as e:
            logger.error(f"Error displaying frame info: {e}", exc_info=True)
            text_widget.setPlainText(f"Error displaying frame info: {e}")

    def _display_differences(
        self,
        frame1: StandardMPCFrame,
        frame2: StandardMPCFrame,
    ) -> None:
        """Display path and device-count changes between two frames."""
        try:
            mpc1 = int(frame1.num_paths or 0)
            mpc2 = int(frame2.num_paths or 0)
            tx1, tx2 = frame1.num_tx, frame2.num_tx
            rx1, rx2 = frame1.num_rx, frame2.num_rx
            diff_text = (
                f"MPC Count: {mpc1} -> {mpc2} (delta {mpc2 - mpc1})\n"
                f"TX Count: {tx1} -> {tx2} (delta {tx2 - tx1})\n"
                f"RX Count: {rx1} -> {rx2} (delta {rx2 - rx1})\n"
            )

            self.diff_info.setPlainText(diff_text)
        except (KeyError, AttributeError, TypeError) as e:
            logger.error(f"Error calculating differences: {e}", exc_info=True)
            self.diff_info.setPlainText(f"Error calculating differences: {e}")


class FrameTimelineWidget(QWidget):
    """Draw frame availability states across a compact timeline."""

    def __init__(self, parent=None):
        """Initialize timeline state and the live-gRPC legend tooltip."""
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setMaximumHeight(200)
        self.frame_states: Dict[int, str] = {}
        self.current_frame = -1
        self.frame_range = (0, 100)  # (min, max) frame indices to display

        self.setToolTip(
            "Frame Timeline:\n"
            "• Green (Computed): Frame is available on server (not in local buffer)\n"
            "• Blue (Buffered): Frame is in local buffer and ready to display\n"
            "• Yellow (Requested): Frame request sent, waiting for server response\n"
            "• Red (Failed): Frame request failed or timed out"
        )

    def set_frame_states(self, states: Dict[int, str], current_frame: int = -1):
        """Update frame states and auto-pad the visible frame range."""
        self.frame_states = states
        self.current_frame = current_frame
        if states:
            all_frames = list(states.keys())
            if all_frames:
                min_frame = min(all_frames)
                max_frame = max(all_frames)
                padding = max(10, (max_frame - min_frame) // 10)
                self.frame_range = (max(0, min_frame - padding), max_frame + padding)
        self.update()

    def paintEvent(self, event):
        """Paint frame-state bars, current-frame highlight, and legend."""
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)

            width = self.width()
            height = self.height()

            painter.fillRect(0, 0, width, height, QColor(245, 245, 245))

            if not self.frame_states or self.frame_range[1] <= self.frame_range[0]:
                painter.setPen(QColor(150, 150, 150))
                painter.drawText(QRect(0, 0, width, height), Qt.AlignCenter, "No frame data")
                return

            colors = {
                "computed": QColor(76, 175, 80),  # Green
                "buffered": QColor(33, 150, 243),  # Blue
                "requested": QColor(255, 193, 7),  # Yellow
                "failed": QColor(244, 67, 54),  # Red
            }

            frame_span = self.frame_range[1] - self.frame_range[0]
            if frame_span == 0:
                return

            # Draw frame indicators
            # Reserve space: 25px for x-axis labels, 20px for legend = 45px total
            bar_height = height - 45
            bar_width = max(
                4, min(8, width // max(1, len(self.frame_states)) * 2)
            )  # Adaptive width, 4-8px
            for frame_idx, state in self.frame_states.items():
                if frame_idx < self.frame_range[0] or frame_idx > self.frame_range[1]:
                    continue

                x_ratio = (frame_idx - self.frame_range[0]) / frame_span
                x = int(x_ratio * width) - bar_width // 2

                color = colors.get(state, QColor(200, 200, 200))
                painter.fillRect(x, 10, bar_width, bar_height, color)

                painter.setPen(QPen(color.darker(120), 1))
                painter.drawRect(x, 10, bar_width, bar_height)

                # Highlight current frame with thicker border
                if frame_idx == self.current_frame:
                    painter.setPen(QPen(QColor(0, 0, 0), 3))
                    painter.drawRect(x - 2, 8, bar_width + 4, bar_height + 4)

            # Draw frame number labels at intervals (x-axis)
            painter.setPen(QColor(100, 100, 100))
            painter.setFont(QFont("Arial", 9))
            label_interval = max(1, frame_span // 10)  # Show ~10 labels
            x_axis_y = height - 35  # Position x-axis labels above legend
            for i in range(self.frame_range[0], self.frame_range[1] + 1, label_interval):
                x_ratio = (i - self.frame_range[0]) / frame_span
                x = int(x_ratio * width)
                painter.drawText(x - 15, x_axis_y, 30, 15, Qt.AlignCenter, str(i))

            # Draw legend with better descriptions (positioned at bottom, below x-axis)
            legend_y = height - 18
            legend_x = 5
            legend_items = [
                ("computed", "Computed (on server)"),
                ("buffered", "Buffered (local)"),
                ("requested", "Requested (pending)"),
                ("failed", "Failed"),
            ]
            painter.setFont(QFont("Arial", 8))
            for state, label in legend_items:
                color = colors.get(state, QColor(200, 200, 200))
                painter.fillRect(legend_x, legend_y, 12, 12, color)
                painter.setPen(QPen(color.darker(120), 1))
                painter.drawRect(legend_x, legend_y, 12, 12)
                painter.setPen(QColor(50, 50, 50))
                text_width = painter.fontMetrics().horizontalAdvance(label)
                painter.drawText(
                    legend_x + 15,
                    legend_y,
                    text_width + 5,
                    12,
                    Qt.AlignLeft | Qt.AlignVCenter,
                    label,
                )
                legend_x += text_width + 25
        finally:
            painter.end()
