"""Connection-status subsection for live-gRPC data sources.

Displays provider endpoint, health, flow-control, and backlog metrics supplied
by ``GrpcProvider.get_connection_status()``.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from shared.logging import get_logger

from ..ui_theme import EMPTY_DISPLAY_VALUE, configure_label, set_widget_role

logger = get_logger("orchav.connection_section")


class ConnectionStatusSection:
    """Connection information and health metrics for live gRPC mode.

    Args:
        parent: The visualizer instance.
        widgets: Shared widget registry dict from DataSourcePanel.
    """

    def __init__(self, parent: Any, widgets: Dict[str, Any]) -> None:
        """Store parent access and the shared DataSourcePanel widget registry."""
        self.parent = parent
        self.widgets = widgets

    def create_content(self) -> QWidget:
        """Create the connection status content widget.

        Returns:
            A QWidget containing connection info and health metrics groups.
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._create_connection_info_group())
        layout.addWidget(self._create_health_metrics_group())

        return container

    # -- UI Construction ------------------------------------------------------

    def _create_connection_info_group(self) -> QGroupBox:
        """Create endpoint, status, uptime, stream, and epoch labels."""
        group = QGroupBox("Connection Information")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)

        for label_text, widget_key, extra_style in [
            ("Endpoint:", "online_endpoint", "font-size: 10px;"),
            ("Status:", "online_status", "font-weight: bold;"),
            ("Uptime:", "online_uptime", ""),
            ("Streaming:", "online_streaming", ""),
            ("Epoch:", "live_grpc_generation_epoch", ""),
        ]:
            row = QHBoxLayout()
            label = QLabel(label_text)
            configure_label(label, role="secondary", min_width=100)
            row.addWidget(label)

            value = QLabel(EMPTY_DISPLAY_VALUE)
            if "font-size" in extra_style:
                configure_label(value, font_size=10)
            if "font-weight: bold" in extra_style:
                configure_label(value, bold=True)
            if widget_key == "online_endpoint":
                value.setWordWrap(True)
                row.addWidget(value, 1)
            else:
                row.addWidget(value)
                row.addStretch()

            self.widgets[widget_key] = value
            layout.addLayout(row)

        return group

    def _create_health_metrics_group(self) -> QGroupBox:
        """Create provider health, backlog, and flow-control labels."""
        group = QGroupBox("Health Metrics")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)

        for label_text, widget_key, extra_style in [
            ("Last Error:", "online_last_error", "font-size: 10px;"),
            ("Queue Depth:", "online_queue_depth", ""),
            ("Pending Frames:", "online_pending_requests", ""),
            ("Ack Backlog:", "online_pending_acks", ""),
            ("Buffer Credits:", "online_buffer_credits", ""),
            ("Stale Drops:", "live_grpc_stale_drops", ""),
            ("Flow Control:", "online_last_flow_control", "font-size: 10px;"),
        ]:
            row = QHBoxLayout()
            label = QLabel(label_text)
            configure_label(label, role="secondary", min_width=100)
            row.addWidget(label)

            value = QLabel(EMPTY_DISPLAY_VALUE)
            if widget_key == "online_last_error":
                configure_label(value, role="error", font_size=10)
            elif "font-size" in extra_style:
                configure_label(value, font_size=10)
            if widget_key == "online_last_error":
                value.setWordWrap(True)
                row.addWidget(value, 1)
            else:
                row.addWidget(value)
                row.addStretch()

            self.widgets[widget_key] = value
            layout.addLayout(row)

        return group

    # -- Update ---------------------------------------------------------------

    def update(self, status: Dict[str, Any]) -> None:
        """Update connection information and health metrics.

        Args:
            status: Connection status dict from ``provider.get_connection_status()``.
        """
        # -- Connection Information ---
        self.widgets["online_endpoint"].setText(status.get("endpoint", EMPTY_DISPLAY_VALUE))

        if status.get("connected", False):
            self.widgets["online_status"].setText("Connected")
            set_widget_role(self.widgets["online_status"], "success")
        else:
            self.widgets["online_status"].setText("Disconnected")
            set_widget_role(self.widgets["online_status"], "error")

        uptime = status.get("connection_uptime", 0.0)
        if uptime > 0:
            if uptime < 60:
                self.widgets["online_uptime"].setText(f"{uptime:.1f} s")
            elif uptime < 3600:
                self.widgets["online_uptime"].setText(f"{uptime / 60:.1f} min")
            else:
                self.widgets["online_uptime"].setText(f"{uptime / 3600:.1f} h")
        else:
            self.widgets["online_uptime"].setText(EMPTY_DISPLAY_VALUE)

        streaming = "Active" if status.get("streaming_active", False) else "Inactive"
        self.widgets["online_streaming"].setText(streaming)
        self.widgets["live_grpc_generation_epoch"].setText(str(status.get("generation_epoch", 0)))

        # -- Health Metrics ---
        last_error = status.get("last_error")
        if last_error:
            set_widget_role(self.widgets["online_last_error"], "error")
            error_time = status.get("last_error_timestamp")
            if error_time:
                elapsed = time.time() - error_time
                if elapsed < 60:
                    error_str = f"{last_error[:50]} ({elapsed:.0f}s ago)"
                else:
                    error_str = f"{last_error[:50]} ({elapsed / 60:.0f}m ago)"
            else:
                error_str = last_error[:80]
            self.widgets["online_last_error"].setText(error_str)
        else:
            self.widgets["online_last_error"].setText("None")
            set_widget_role(self.widgets["online_last_error"], "success")

        self.widgets["online_queue_depth"].setText(str(status.get("request_queue_depth", 0)))

        pending_requests = status.get("pending_request_count")
        if pending_requests is not None:
            self.widgets["online_pending_requests"].setText(str(pending_requests))

        pending_acks = status.get("pending_ack_count")
        if pending_acks is not None:
            self.widgets["online_pending_acks"].setText(str(pending_acks))

        buffer_credits = status.get("buffer_free_capacity")
        if buffer_credits is not None:
            max_size = status.get("prefetch_window", 0)
            if max_size:
                self.widgets["online_buffer_credits"].setText(f"{buffer_credits} / {max_size}")
            else:
                self.widgets["online_buffer_credits"].setText(str(buffer_credits))

        self.widgets["live_grpc_stale_drops"].setText(str(status.get("stale_frames_dropped", 0)))

        last_flow = status.get("last_flow_control_at")
        if last_flow:
            delta = time.time() - last_flow
            if delta < 1.0:
                flow_text = "just now"
            elif delta < 60.0:
                flow_text = f"{delta:.0f}s ago"
            else:
                flow_text = f"{delta / 60:.1f}m ago"
            self.widgets["online_last_flow_control"].setText(flow_text)
        else:
            self.widgets["online_last_flow_control"].setText(EMPTY_DISPLAY_VALUE)

    def set_defaults(self) -> None:
        """Reset all connection/health widgets to default values."""
        for key in [
            "online_endpoint",
            "online_status",
            "online_uptime",
            "online_streaming",
            "live_grpc_generation_epoch",
            "online_last_error",
            "online_queue_depth",
            "online_pending_requests",
            "online_pending_acks",
            "online_buffer_credits",
            "live_grpc_stale_drops",
            "online_last_flow_control",
        ]:
            widget = self.widgets.get(key)
            if widget is not None:
                widget.setText(EMPTY_DISPLAY_VALUE)
                if key in {"online_status", "online_last_error"}:
                    set_widget_role(widget, None)
