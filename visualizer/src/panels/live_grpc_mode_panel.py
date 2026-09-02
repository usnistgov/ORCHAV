"""Live-gRPC data-source controls composed from focused subsections.

Thin orchestrator that delegates to three focused sub-sections:
- ConnectionStatusSection: connection info and health metrics
- StreamingControlSection: buffer status, frame timeline, prefetch, actions
- PerformanceSection: telemetry, performance graphs, and connection diagnostics
"""

from __future__ import annotations

from typing import Any, Dict

from PySide6.QtWidgets import QVBoxLayout, QWidget

from shared.logging import get_logger

from .base import BasePanel
from .data_source.connection_section import ConnectionStatusSection
from .data_source.performance_section import PerformanceSection
from .data_source.streaming_section import StreamingControlSection

logger = get_logger("orchav.live_grpc_mode_panel")


class LiveGrpcModePanel(BasePanel):
    """Compose live-gRPC status, streaming, and diagnostics.

    This panel delegates UI construction, update logic, and event handlers
    to three focused sub-sections. It shares a ``widgets`` dict with the
    parent ``DataSourcePanel`` so that both panels can read each other's
    widgets when needed.

    Args:
        parent_widget: The visualizer (or host) that owns the frame source.
        widgets: Shared widget registry dict from ``DataSourcePanel``.
        button_style_fn: Callable returning button stylesheet string.
    """

    def __init__(
        self,
        parent_widget: Any,
        widgets: Dict[str, Any],
        button_style_fn: Any,
    ) -> None:
        """Initialize subsection objects that share DataSourcePanel widgets."""
        super().__init__(parent_widget)
        self.widgets = widgets
        self._get_button_style = button_style_fn

        self._connection = ConnectionStatusSection(parent_widget, widgets)
        self._streaming = StreamingControlSection(parent_widget, widgets, button_style_fn)
        self._performance = PerformanceSection(parent_widget, widgets, button_style_fn)

    # UI Construction

    def create_content(self) -> QWidget:
        """Create live gRPC mode content widget with all sub-sections.

        Returns:
            A QWidget container holding every live gRPC sub-section.
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._connection.create_content())
        layout.addWidget(self._performance.create_content())
        layout.addWidget(self._streaming.create_content())
        return container

    def cleanup(self) -> None:
        """Release subsection subscriptions owned outside the widget tree."""
        self._performance.cleanup()

    def update_live_grpc_mode_info(self) -> None:
        """Refresh live-gRPC subsections from the current GrpcProvider."""
        try:
            frame_source = self.parent.frame_source
            if not hasattr(frame_source, "provider") or frame_source.provider is None:
                frame_source.open()

            provider = frame_source.provider
            from ..io.grpc_provider import GrpcProvider

            if not isinstance(provider, GrpcProvider):
                self.set_live_grpc_default_values()
                return

            if hasattr(provider, "streaming_buffer") and "prefetch_lookahead" in self.widgets:
                ui_lookahead = self.widgets["prefetch_lookahead"].value()
                if provider.streaming_buffer.lookahead != ui_lookahead:
                    provider.set_prefetch_lookahead(ui_lookahead)
                    logger.debug(
                        "[PREFETCH] Applied UI lookahead=%d to provider at initialization",
                        ui_lookahead,
                    )

            status = provider.get_connection_status()
            buffer_status = provider.get_buffer_status()
            latency_stats = provider.get_latency_statistics()
            request_history = provider.get_request_history(limit=100)

            self._connection.update(status)
            self._streaming.update(provider, buffer_status, request_history)
            self._performance.update(
                provider, status, buffer_status, latency_stats, request_history
            )

        except (KeyError, AttributeError, ValueError, TypeError, OSError) as e:
            logger.error("Error updating live gRPC mode info: %s", e)
            self.set_live_grpc_default_values()

    def set_live_grpc_default_values(self) -> None:
        """Reset live-gRPC subsection widgets when no provider is available."""
        self._connection.set_defaults()
        self._streaming.set_defaults()
        self._performance.set_defaults()
