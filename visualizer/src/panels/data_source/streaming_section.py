"""Streaming/buffer subsection for live-gRPC data sources.

Shows local buffer state, provider/server frame availability, smart prefetch
settings, and connection actions that operate on ``LiveGrpcSource``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from shared.logging import get_logger

from ..collapsible_section import CollapsibleSection
from ..ui_theme import EMPTY_DISPLAY_VALUE, compact_text_edit_style, configure_label
from .widgets import FrameComparisonDialog, FrameTimelineWidget

logger = get_logger("orchav.streaming_section")


class StreamingControlSection:
    """Buffer status, frame timeline, prefetch controls, and actions.

    Args:
        parent: The visualizer instance.
        widgets: Shared widget registry dict from DataSourcePanel.
        button_style_fn: Callable returning QPushButton stylesheet string.
    """

    def __init__(
        self,
        parent: Any,
        widgets: Dict[str, Any],
        button_style_fn: Callable[[], str],
    ) -> None:
        """Store parent access, shared widgets, and button styling callback."""
        self.parent = parent
        self.widgets = widgets
        self._get_button_style = button_style_fn

    def create_content(self) -> QWidget:
        """Create streaming control content widget.

        Returns:
            A QWidget containing buffer, timeline, prefetch, and action groups.
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._create_buffer_status_group())
        layout.addWidget(self._create_frame_timeline_group())
        layout.addWidget(self._create_prefetching_controls_group())
        layout.addWidget(self._create_online_actions_group())

        return container

    # -- Buffer Status --------------------------------------------------------

    def _create_buffer_status_group(self) -> QGroupBox:
        """Create labels for local buffer capacity and available frame IDs."""
        group = QGroupBox("Buffer Status")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)

        for label_text, widget_key in [
            ("Buffer Size:", "online_buffer_size"),
            ("Utilization:", "online_buffer_util"),
            ("Latest Frame:", "online_latest_frame"),
            ("Available:", "online_available_frames"),
        ]:
            row = QHBoxLayout()
            label = QLabel(label_text)
            configure_label(label, role="secondary", min_width=100)
            row.addWidget(label)
            value = QLabel(EMPTY_DISPLAY_VALUE)
            self.widgets[widget_key] = value
            row.addWidget(value)
            row.addStretch()
            layout.addLayout(row)

        frames_section = CollapsibleSection("Buffered Frame IDs", start_open=False)
        frames_layout = frames_section.content_layout()

        self.widgets["online_buffered_frames_list"] = QTextEdit()
        self.widgets["online_buffered_frames_list"].setReadOnly(True)
        self.widgets["online_buffered_frames_list"].setMaximumHeight(100)
        self.widgets["online_buffered_frames_list"].setStyleSheet(
            compact_text_edit_style(monospace=True)
        )
        frames_layout.addWidget(self.widgets["online_buffered_frames_list"])
        layout.addWidget(frames_section)

        return group

    # -- Frame Timeline -------------------------------------------------------

    def _create_frame_timeline_group(self) -> QGroupBox:
        """Create the timeline that merges server, pending, and local states."""
        group = QGroupBox("Frame Timeline")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)

        self.widgets["frame_timeline"] = FrameTimelineWidget()
        layout.addWidget(self.widgets["frame_timeline"])

        return group

    # -- Smart Prefetching Controls -------------------------------------------

    def _create_prefetching_controls_group(self) -> QGroupBox:
        """Create client/server lookahead and pause-policy controls."""
        group = QGroupBox("Smart Prefetching")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        self.widgets["prefetch_enable"] = QCheckBox("Enable Auto-Prefetch")
        self.widgets["prefetch_enable"].setChecked(True)
        self.widgets["prefetch_enable"].setToolTip(
            "Automatically prefetch frames based on playback direction"
        )
        self.widgets["prefetch_enable"].stateChanged.connect(self._on_prefetch_enable_changed)
        layout.addWidget(self.widgets["prefetch_enable"])

        lookahead_row = QHBoxLayout()
        lookahead_label = QLabel("Lookahead:")
        configure_label(lookahead_label, role="secondary", min_width=100)
        lookahead_row.addWidget(lookahead_label)
        self.widgets["prefetch_lookahead"] = QSpinBox()
        self.widgets["prefetch_lookahead"].setRange(1, 100)
        self.widgets["prefetch_lookahead"].setValue(25)
        self.widgets["prefetch_lookahead"].setToolTip(
            "Number of frames to prefetch ahead (client-side) and compute ahead (server-side)"
        )
        self.widgets["prefetch_lookahead"].valueChanged.connect(self._on_prefetch_lookahead_changed)
        lookahead_row.addWidget(self.widgets["prefetch_lookahead"])
        lookahead_row.addStretch()
        layout.addLayout(lookahead_row)

        self.widgets["prefetch_pause_when_paused"] = QCheckBox(
            "Pause Prefetch When Playback Paused"
        )
        self.widgets["prefetch_pause_when_paused"].setChecked(False)
        self.widgets["prefetch_pause_when_paused"].setToolTip(
            "Stop prefetching when animation is paused"
        )
        self.widgets["prefetch_pause_when_paused"].stateChanged.connect(
            self._on_prefetch_pause_changed
        )
        layout.addWidget(self.widgets["prefetch_pause_when_paused"])

        self.widgets["prefetch_status"] = QLabel("Prefetch: Active")
        configure_label(self.widgets["prefetch_status"], role="secondary", font_size=9, italic=True)
        layout.addWidget(self.widgets["prefetch_status"])

        return group

    # -- Online Actions -------------------------------------------------------

    def _create_online_actions_group(self) -> QGroupBox:
        """Create live-gRPC connection and buffer action buttons."""
        group = QGroupBox("Actions")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        button_row = QHBoxLayout()

        reconnect_btn = QPushButton("Reconnect")
        reconnect_btn.setStyleSheet(self._get_button_style())
        reconnect_btn.clicked.connect(self._on_reconnect_clicked)
        self.widgets["online_reconnect_btn"] = reconnect_btn
        button_row.addWidget(reconnect_btn)

        clear_btn = QPushButton("Reset Live Frames")
        clear_btn.setStyleSheet(self._get_button_style())
        clear_btn.setToolTip(
            "Discard client and generator frame caches; requested frames will be recomputed"
        )
        clear_btn.clicked.connect(self._on_clear_buffer_clicked)
        self.widgets["online_clear_btn"] = clear_btn
        button_row.addWidget(clear_btn)

        self.widgets["online_pause_resume_btn"] = QPushButton("Pause Streaming")
        self.widgets["online_pause_resume_btn"].setStyleSheet(self._get_button_style())
        self.widgets["online_pause_resume_btn"].clicked.connect(self._on_pause_resume_clicked)
        self.widgets["online_pause_resume_btn"].setCheckable(True)
        button_row.addWidget(self.widgets["online_pause_resume_btn"])

        compare_btn = QPushButton("Compare Frames")
        compare_btn.setStyleSheet(self._get_button_style())
        compare_btn.clicked.connect(self._on_compare_frames_clicked)
        button_row.addWidget(compare_btn)

        button_row.addStretch()
        layout.addLayout(button_row)

        buffer_size_row = QHBoxLayout()
        buffer_size_label = QLabel("Buffer Size:")
        configure_label(buffer_size_label, role="secondary", min_width=100)
        buffer_size_row.addWidget(buffer_size_label)

        self.widgets["buffer_size_slider"] = QSlider(Qt.Horizontal)
        self.widgets["buffer_size_slider"].setRange(5, 200)
        self.widgets["buffer_size_slider"].setValue(50)
        self.widgets["buffer_size_slider"].setToolTip("Adjust streaming buffer size")
        self.widgets["buffer_size_slider"].valueChanged.connect(self._on_buffer_size_changed)
        buffer_size_row.addWidget(self.widgets["buffer_size_slider"], 1)

        self.widgets["buffer_size_label"] = QLabel("50")
        configure_label(self.widgets["buffer_size_label"], min_width=30)
        self.widgets["buffer_size_label"].setAlignment(Qt.AlignRight)
        buffer_size_row.addWidget(self.widgets["buffer_size_label"])

        layout.addLayout(buffer_size_row)

        return group

    # -- Update ---------------------------------------------------------------

    def update(
        self,
        provider: Any,
        buffer_status: Dict[str, Any],
        request_history: List[Dict[str, Any]],
    ) -> None:
        """Update buffer status, frame timeline, and prefetch status.

        Args:
            provider: The GrpcProvider instance.
            buffer_status: Buffer status dict from ``provider.get_buffer_status()``.
            request_history: Recent request history entries.
        """
        self._update_buffer_status(buffer_status)
        self._update_frame_timeline(provider, buffer_status, request_history)
        self._update_prefetch_status(provider)

    def _update_buffer_status(self, buffer_status: Dict[str, Any]) -> None:
        """Update buffer status labels."""
        buffer_size = buffer_status.get("buffer_size", 0)
        max_size = buffer_status.get("max_buffer_size", 0)
        utilization = (buffer_size / max_size * 100.0) if max_size > 0 else 0.0

        self.widgets["online_buffer_size"].setText(f"{buffer_size} / {max_size}")
        self.widgets["online_buffer_util"].setText(f"{utilization:.1f}%")

        slider = self.widgets.get("buffer_size_slider")
        slider_label = self.widgets.get("buffer_size_label")
        if slider is not None:
            with QSignalBlocker(slider):
                slider.setValue(max_size if max_size > 0 else slider.minimum())
        if slider_label is not None:
            slider_label.setText(str(max_size))

        latest_frame = buffer_status.get("latest_frame")
        if latest_frame is not None:
            self.widgets["online_latest_frame"].setText(str(latest_frame))
        else:
            self.widgets["online_latest_frame"].setText(EMPTY_DISPLAY_VALUE)

        available_frames = buffer_status.get("available_frames", [])
        if available_frames:
            sorted_frames = sorted(available_frames)
            if len(sorted_frames) == 1:
                summary = f"1 frame ({sorted_frames[0]})"
            else:
                summary = f"{len(sorted_frames)} frames ({sorted_frames[0]}-{sorted_frames[-1]})"
        else:
            sorted_frames = []
            summary = "0 frames"
        self.widgets["online_available_frames"].setText(summary)

        buffered_list = self.widgets.get("online_buffered_frames_list")
        if buffered_list is not None and buffered_list.isVisible():
            if len(sorted_frames) > 50:
                frames_text = (
                    f"{', '.join(map(str, sorted_frames[:25]))} "
                    f"... {', '.join(map(str, sorted_frames[-25:]))}"
                )
            else:
                frames_text = ", ".join(map(str, sorted_frames))
            buffered_list.setPlainText(frames_text if sorted_frames else "(no frames buffered)")

    def _update_frame_timeline(
        self,
        provider: Any,
        buffer_status: Dict[str, Any],
        request_history: List[Dict[str, Any]],
    ) -> None:
        """Merge provider internals and request history into timeline states."""
        if "frame_timeline" not in self.widgets:
            return

        try:
            current_frame = getattr(self.parent, "animation_step", -1)
            frame_states: Dict[int, str] = {}

            # Server-available frames
            server_available_frames: set = set()
            if hasattr(provider, "_get_frame_info"):
                frame_info = provider._get_frame_info()
                if frame_info and "available_frames" in frame_info:
                    server_available_frames = set(frame_info["available_frames"])

            # Pending requests
            pending_frames: set = {
                request.frame_idx
                for request in getattr(provider, "get_pending_frame_requests", lambda: [])()
            }

            # Local buffered frames
            local_buffered_frames = set(buffer_status.get("available_frames", []))

            # Failed frames from request history
            failed_frames: set = set()
            for req in request_history[:100]:
                frame_idx = req.get("frame_idx", -1)
                if frame_idx < 0:
                    continue
                if not req.get("success", False) and frame_idx not in pending_frames:
                    failed_frames.add(frame_idx)

            # Build final state map with priority
            for frame_idx in failed_frames:
                if frame_idx not in pending_frames and frame_idx not in local_buffered_frames:
                    frame_states[frame_idx] = "failed"
            for frame_idx in pending_frames:
                frame_states[frame_idx] = "requested"
            for frame_idx in local_buffered_frames:
                if frame_idx not in frame_states:
                    frame_states[frame_idx] = "buffered"
            for frame_idx in server_available_frames:
                if frame_idx not in frame_states:
                    frame_states[frame_idx] = "computed"

            self.widgets["frame_timeline"].set_frame_states(frame_states, current_frame)

        except (KeyError, AttributeError, ValueError) as e:
            logger.debug("Error updating frame timeline: %s", e)

    def _update_prefetch_status(self, provider: Any) -> None:
        """Update smart prefetching status based on playback state."""
        if "prefetch_status" not in self.widgets:
            return

        try:
            prefetch_enabled = self.widgets.get("prefetch_enable")
            if not prefetch_enabled or not prefetch_enabled.isChecked():
                self.widgets["prefetch_status"].setText("Prefetch: Disabled")
                return

            pause_when_paused = self.widgets.get("prefetch_pause_when_paused")
            animation_running = getattr(self.parent, "animation_running", False)
            if pause_when_paused and pause_when_paused.isChecked() and not animation_running:
                self.widgets["prefetch_status"].setText("Prefetch: Paused (playback paused)")
                return

            direction = getattr(self.parent, "play_direction", 1)
            direction_text = "Backward" if direction < 0 else "Forward"
            self.widgets["prefetch_status"].setText(f"Prefetch: Active ({direction_text})")
        except (KeyError, AttributeError) as e:
            logger.debug("Error updating prefetch status: %s", e)

    def set_defaults(self) -> None:
        """Reset all streaming widgets to default values."""
        for key in [
            "online_buffer_size",
            "online_buffer_util",
            "online_latest_frame",
            "online_available_frames",
        ]:
            widget = self.widgets.get(key)
            if widget is not None:
                widget.setText(EMPTY_DISPLAY_VALUE)
        buffered_list = self.widgets.get("online_buffered_frames_list")
        if buffered_list is not None:
            buffered_list.setPlainText("(no frames buffered)")

    # -- Event Handlers -------------------------------------------------------

    def _on_reconnect_clicked(self) -> None:
        """Handle reconnect button click."""
        try:
            from ...io.frame_sources import LiveGrpcSource

            if isinstance(self.parent.frame_source, LiveGrpcSource):
                provider = self.parent.frame_source.provider
                if provider:
                    provider.close()
                    provider.open()
                    logger.info("Reconnected to gRPC server")
        except (OSError, RuntimeError) as e:
            logger.error("Error reconnecting: %s", e)

    def _on_clear_buffer_clicked(self) -> None:
        """Discard client/server live frames so later requests recompute them."""
        try:
            from ...io.frame_sources import LiveGrpcSource

            if isinstance(self.parent.frame_source, LiveGrpcSource):
                self.parent.frame_source.request_cache_flush(reason="UI Reset Live Frames")
                logger.info("Requested live frame reset (client + generator)")
        except (OSError, RuntimeError) as e:
            logger.error("Error clearing buffer: %s", e)

    def _on_pause_resume_clicked(self) -> None:
        """Handle pause/resume streaming button click."""
        try:
            from ...io.frame_sources import LiveGrpcSource

            if isinstance(self.parent.frame_source, LiveGrpcSource):
                provider = self.parent.frame_source.provider
                if provider:
                    is_paused = self.widgets["online_pause_resume_btn"].isChecked()
                    if is_paused:
                        provider.pause_streaming()
                        self.widgets["online_pause_resume_btn"].setText("Resume Streaming")
                        logger.info("Paused streaming")
                    else:
                        provider.resume_streaming()
                        self.widgets["online_pause_resume_btn"].setText("Pause Streaming")
                        logger.info("Resumed streaming")
        except (OSError, RuntimeError) as e:
            logger.error("Error pausing/resuming stream: %s", e)

    def _on_buffer_size_changed(self, value: int) -> None:
        """Handle buffer size slider change."""
        try:
            self.widgets["buffer_size_label"].setText(str(value))

            from ...io.frame_sources import LiveGrpcSource

            if isinstance(self.parent.frame_source, LiveGrpcSource):
                provider = self.parent.frame_source.provider
                if provider:
                    provider.set_buffer_size(value)
                    logger.info("Changed buffer size to %d", value)
        except (OSError, RuntimeError, ValueError) as e:
            logger.error("Error changing buffer size: %s", e)

    def _on_compare_frames_clicked(self) -> None:
        """Handle compare frames button click."""
        try:
            dialog = FrameComparisonDialog(self.parent)
            dialog.exec()
        except (RuntimeError, ValueError) as e:
            logger.error("Error opening frame comparison dialog: %s", e)
            QMessageBox.warning(self.parent, "Error", f"Failed to open frame comparison: {e}")

    def _on_prefetch_enable_changed(self, state: int) -> None:
        """Apply the live prefetch policy after the checkbox changes."""
        try:
            enabled = state == Qt.Checked
            if "prefetch_status" in self.widgets:
                status_text = "Prefetch: Active" if enabled else "Prefetch: Disabled"
                self.widgets["prefetch_status"].setText(status_text)
            logger.info("Prefetch enabled: %s", enabled)
            self._sync_prefetch_policy()
        except (OSError, RuntimeError, AttributeError) as e:
            logger.error("Error changing prefetch enable: %s", e)

    def _on_prefetch_lookahead_changed(self, value: int) -> None:
        """Handle prefetch lookahead spinbox change."""
        try:
            from ...io.frame_sources import LiveGrpcSource

            if isinstance(self.parent.frame_source, LiveGrpcSource):
                provider = self.parent.frame_source.provider
                set_lookahead = getattr(provider, "set_prefetch_lookahead", None)
                if callable(set_lookahead):
                    set_lookahead(value)
                    logger.info("Prefetch lookahead changed to %d", value)
        except (OSError, RuntimeError, AttributeError) as e:
            logger.error("Error changing prefetch lookahead: %s", e)

    def _on_prefetch_pause_changed(self, _state: int) -> None:
        """Apply the pause-dependent prefetch policy immediately."""
        self._sync_prefetch_policy()

    def _sync_prefetch_policy(self) -> None:
        """Delegate provider updates to the playback-policy owner."""
        controller = getattr(self.parent, "animation_controller", None)
        sync = getattr(controller, "sync_prefetch_settings", None)
        if callable(sync):
            sync()
