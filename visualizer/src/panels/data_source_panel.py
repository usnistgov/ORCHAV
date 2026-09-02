"""Data-source status panel for file, live gRPC, and remote-HDF5 modes.

The panel owns the shared widget registry and switches between mode-specific
summaries. File and live-gRPC paths may open their providers to report status;
remote-HDF5 status stays passive so viewing the panel does not initiate a
network connection.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from shared.frames.providers import Hdf5Provider
from shared.logging import get_logger

from ..app.theme import current_theme
from .base import BasePanel
from .live_grpc_mode_panel import LiveGrpcModePanel
from .ui_theme import EMPTY_DISPLAY_VALUE, compact_button_style, configure_label, set_widget_role

logger = get_logger("orchav.data_source_panel")

INITIAL_STATUS_DELAY_MS = 100
SECONDARY_STATUS_DELAY_MS = 1000
STATUS_UPDATE_INTERVAL_MS = 2000
EMPTY_VALUE = EMPTY_DISPLAY_VALUE


class DataSourcePanel(BasePanel):
    """Build mode-specific status summaries for the active frame source."""

    def __init__(self, parent_widget: Any) -> None:
        """Initialize the shared live-gRPC subpanel and widget registry."""
        super().__init__(parent_widget)
        self._live_grpc_panel = LiveGrpcModePanel(
            parent_widget, self.widgets, self._get_button_style
        )

    def create_panel(self) -> Any:
        """Create all mode containers and start lightweight status refreshes."""
        group = self.create_group_box("Data Source")

        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        layout.addLayout(self._create_mode_indicator())

        self._file_mode_container = self._create_file_mode_content()
        self._live_grpc_mode_container = self._live_grpc_panel.create_content()
        self._remote_hdf5_mode_container = self._create_remote_hdf5_mode_content()

        layout.addWidget(self._file_mode_container)
        layout.addWidget(self._live_grpc_mode_container)
        layout.addWidget(self._remote_hdf5_mode_container)

        # Providers may attach after panel construction during startup wiring.
        self._schedule_status_refresh(group, INITIAL_STATUS_DELAY_MS)
        self._schedule_status_refresh(group, SECONDARY_STATUS_DELAY_MS)

        self._update_timer = QTimer(group)
        self._update_timer.setInterval(STATUS_UPDATE_INTERVAL_MS)
        self._update_timer.timeout.connect(self._update_status)

        return group

    def cleanup(self) -> None:
        """Stop polling and release live-gRPC subsection subscriptions."""
        timer = getattr(self, "_update_timer", None)
        if timer is not None:
            timer.stop()
        self._live_grpc_panel.cleanup()

    def _schedule_status_refresh(self, owner: QWidget, delay_ms: int) -> None:
        """Run a one-shot refresh while the panel widget is still alive."""
        timer = QTimer(owner)
        timer.setSingleShot(True)
        timer.timeout.connect(self._update_status)
        timer.timeout.connect(timer.deleteLater)
        timer.start(delay_ms)

    def _create_mode_indicator(self) -> QHBoxLayout:
        """Create the badge that identifies the current frame-source mode."""
        row = QHBoxLayout()
        row.setSpacing(8)
        row.setContentsMargins(4, 0, 4, 0)

        mode_label = QLabel("Mode:")
        row.addWidget(mode_label)
        configure_label(mode_label, bold=True)

        self.widgets["mode_badge"] = QLabel("File")
        self._set_mode_badge("File", "file")
        row.addWidget(self.widgets["mode_badge"])

        row.addStretch()
        return row

    # File Mode Content

    def _create_file_mode_content(self) -> QWidget:
        """Create file-mode status fields for local HDF5 frame sources."""
        container = QWidget()
        columns = QHBoxLayout(container)
        columns.setSpacing(12)
        columns.setContentsMargins(0, 0, 0, 0)

        # --- Column 1: Format Information ---
        col1 = QVBoxLayout()
        col1.setSpacing(2)
        col1_header = QLabel("Format")
        configure_label(col1_header, font_size=10, bold=True)
        col1.addWidget(col1_header)

        for field_label, widget_key in [
            ("Format:", "format_type"),
            ("Files:", "file_count"),
            ("Chunks:", "chunk_info"),
        ]:
            row = QHBoxLayout()
            row.setSpacing(4)
            row.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(field_label)
            configure_label(lbl, role="secondary", font_size=10)
            row.addWidget(lbl)
            val = QLabel(EMPTY_VALUE)
            configure_label(val, font_size=10)
            self.widgets[widget_key] = val
            row.addWidget(val)
            row.addStretch()
            col1.addLayout(row)

        # Path row (word-wrappable, gets stretch)
        path_row = QHBoxLayout()
        path_row.setSpacing(4)
        path_row.setContentsMargins(0, 0, 0, 0)
        path_lbl = QLabel("Path:")
        configure_label(path_lbl, role="secondary", font_size=10)
        path_row.addWidget(path_lbl)
        self.widgets["file_path"] = QLabel(EMPTY_VALUE)
        configure_label(self.widgets["file_path"], font_size=10, word_wrap=True)
        path_row.addWidget(self.widgets["file_path"], 1)
        col1.addLayout(path_row)

        col1.addStretch()
        columns.addLayout(col1, 1)

        # --- Column 2: Loading Metrics ---
        col2 = QVBoxLayout()
        col2.setSpacing(2)
        col2_header = QLabel("Loading")
        configure_label(col2_header, font_size=10, bold=True)
        col2.addWidget(col2_header)

        for field_label, widget_key in [
            ("Time:", "load_time"),
            ("Preload:", "preload_status"),
            ("Mode:", "loading_mode"),
            ("Memory:", "memory_usage"),
        ]:
            row = QHBoxLayout()
            row.setSpacing(4)
            row.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(field_label)
            configure_label(lbl, role="secondary", font_size=10)
            row.addWidget(lbl)
            val = QLabel(EMPTY_VALUE)
            configure_label(val, font_size=10)
            self.widgets[widget_key] = val
            row.addWidget(val)
            row.addStretch()
            col2.addLayout(row)

        col2.addStretch()
        columns.addLayout(col2, 1)

        # --- Column 3: Frame Statistics ---
        col3 = QVBoxLayout()
        col3.setSpacing(2)
        col3_header = QLabel("Frames")
        configure_label(col3_header, font_size=10, bold=True)
        col3.addWidget(col3_header)

        for field_label, widget_key in [
            ("Total:", "total_frames"),
            ("Loaded:", "loaded_frames"),
            ("Range:", "frame_range"),
            ("Size:", "file_size"),
        ]:
            row = QHBoxLayout()
            row.setSpacing(4)
            row.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(field_label)
            configure_label(lbl, role="secondary", font_size=10)
            row.addWidget(lbl)
            val = QLabel(EMPTY_VALUE)
            configure_label(val, font_size=10)
            self.widgets[widget_key] = val
            row.addWidget(val)
            row.addStretch()
            col3.addLayout(row)

        col3.addStretch()
        columns.addLayout(col3, 1)

        return container

    # Remote HDF5 Mode Content

    def _create_remote_hdf5_mode_content(self) -> QWidget:
        """Create passive status fields for remote-HDF5 frame sources."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._create_remote_server_info_group())
        layout.addWidget(self._create_remote_cache_status_group())
        layout.addWidget(self._create_remote_frame_stats_group())

        return container

    def _create_remote_server_info_group(self) -> Any:
        """Create remote server metadata/status labels."""
        group = self.create_subgroup_box("Server Connection")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)

        server_row = QHBoxLayout()
        server_label = QLabel("Server:")
        configure_label(server_label, role="secondary", min_width=100)
        server_row.addWidget(server_label)
        self.widgets["remote_server_address"] = QLabel(EMPTY_VALUE)
        configure_label(self.widgets["remote_server_address"], font_size=10, word_wrap=True)
        server_row.addWidget(self.widgets["remote_server_address"], 1)
        layout.addLayout(server_row)

        status_row = QHBoxLayout()
        status_label = QLabel("Status:")
        configure_label(status_label, role="secondary", min_width=100)
        status_row.addWidget(status_label)
        self.widgets["remote_connection_status"] = QLabel(EMPTY_VALUE)
        configure_label(self.widgets["remote_connection_status"], bold=True)
        status_row.addWidget(self.widgets["remote_connection_status"])
        status_row.addStretch()
        layout.addLayout(status_row)

        scene_row = QHBoxLayout()
        scene_label = QLabel("Scene:")
        configure_label(scene_label, role="secondary", min_width=100)
        scene_row.addWidget(scene_label)
        self.widgets["remote_scene_name"] = QLabel(EMPTY_VALUE)
        scene_row.addWidget(self.widgets["remote_scene_name"])
        scene_row.addStretch()
        layout.addLayout(scene_row)

        frame_set_row = QHBoxLayout()
        frame_set_label = QLabel("Frame Set:")
        configure_label(frame_set_label, role="secondary", min_width=100)
        frame_set_row.addWidget(frame_set_label)
        self.widgets["remote_frame_set_id"] = QLabel(EMPTY_VALUE)
        frame_set_row.addWidget(self.widgets["remote_frame_set_id"])
        frame_set_row.addStretch()
        layout.addLayout(frame_set_row)

        snapshot_row = QHBoxLayout()
        snapshot_label = QLabel("Snapshot:")
        configure_label(snapshot_label, role="secondary", min_width=100)
        snapshot_row.addWidget(snapshot_label)
        self.widgets["remote_snapshot_status"] = QLabel(EMPTY_VALUE)
        snapshot_row.addWidget(self.widgets["remote_snapshot_status"])
        snapshot_row.addStretch()
        layout.addLayout(snapshot_row)

        return group

    def _create_remote_cache_status_group(self) -> Any:
        """Create local cache summary labels for remote-HDF5 mode."""
        group = self.create_subgroup_box("Local Cache")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)

        size_row = QHBoxLayout()
        size_label = QLabel("Max Size:")
        configure_label(size_label, role="secondary", min_width=100)
        size_row.addWidget(size_label)
        self.widgets["remote_cache_size"] = QLabel(EMPTY_VALUE)
        size_row.addWidget(self.widgets["remote_cache_size"])
        size_row.addStretch()
        layout.addLayout(size_row)

        cached_row = QHBoxLayout()
        cached_label = QLabel("Cached Frames:")
        configure_label(cached_label, role="secondary", min_width=100)
        cached_row.addWidget(cached_label)
        self.widgets["remote_cached_frames"] = QLabel(EMPTY_VALUE)
        cached_row.addWidget(self.widgets["remote_cached_frames"])
        cached_row.addStretch()
        layout.addLayout(cached_row)

        util_row = QHBoxLayout()
        util_label = QLabel("Utilization:")
        configure_label(util_label, role="secondary", min_width=100)
        util_row.addWidget(util_label)
        self.widgets["remote_cache_utilization"] = QLabel(EMPTY_VALUE)
        util_row.addWidget(self.widgets["remote_cache_utilization"])
        util_row.addStretch()
        layout.addLayout(util_row)

        hit_row = QHBoxLayout()
        hit_label = QLabel("Hit Ratio:")
        configure_label(hit_label, role="secondary", min_width=100)
        hit_row.addWidget(hit_label)
        self.widgets["remote_cache_hit_ratio"] = QLabel(EMPTY_VALUE)
        hit_row.addWidget(self.widgets["remote_cache_hit_ratio"])
        hit_row.addStretch()
        layout.addLayout(hit_row)

        latency_row = QHBoxLayout()
        latency_label = QLabel("Fetch Latency:")
        configure_label(latency_label, role="secondary", min_width=100)
        latency_row.addWidget(latency_label)
        self.widgets["remote_fetch_latency"] = QLabel(EMPTY_VALUE)
        latency_row.addWidget(self.widgets["remote_fetch_latency"])
        latency_row.addStretch()
        layout.addLayout(latency_row)

        return group

    def _create_remote_frame_stats_group(self) -> Any:
        """Create remote frame-count and frame-range labels."""
        group = self.create_subgroup_box("Frame Statistics")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)

        total_row = QHBoxLayout()
        total_label = QLabel("Total Frames:")
        configure_label(total_label, role="secondary", min_width=100)
        total_row.addWidget(total_label)
        self.widgets["remote_total_frames"] = QLabel(EMPTY_VALUE)
        total_row.addWidget(self.widgets["remote_total_frames"])
        total_row.addStretch()
        layout.addLayout(total_row)

        range_row = QHBoxLayout()
        range_label = QLabel("Frame Range:")
        configure_label(range_label, role="secondary", min_width=100)
        range_row.addWidget(range_label)
        self.widgets["remote_frame_range"] = QLabel(EMPTY_VALUE)
        range_row.addWidget(self.widgets["remote_frame_range"])
        range_row.addStretch()
        layout.addLayout(range_row)

        return group

    # Shared Helpers

    def _get_button_style(self) -> str:
        """Return the shared button style used by live-gRPC subsections."""
        return compact_button_style()

    # Status Update Dispatch

    def refresh_source_status(self) -> None:
        """Refresh the active source summary and its live polling policy."""
        self._update_status()

    def _set_live_polling(self, enabled: bool) -> None:
        """Poll source telemetry only while a live gRPC source is active."""
        timer = getattr(self, "_update_timer", None)
        if timer is None:
            return
        if enabled:
            if not timer.isActive():
                timer.start()
        elif timer.isActive():
            timer.stop()

    def _update_status(self) -> None:
        """Dispatch status refreshes based on the active frame-source class."""
        if not hasattr(self.parent, "frame_source") or not self.parent.frame_source:
            self._set_live_polling(False)
            self._set_default_values()
            self._hide_all_mode_containers()
            return

        from ..io.frame_sources import FileSource, LiveGrpcSource, RemoteHdf5Source

        self._hide_all_mode_containers()

        if isinstance(self.parent.frame_source, FileSource):
            self._set_live_polling(False)
            self._set_mode_badge("File", "file")
            self._file_mode_container.setVisible(True)
            self._update_file_mode_info()

        elif isinstance(self.parent.frame_source, LiveGrpcSource):
            self._set_live_polling(True)
            self._set_mode_badge("Live gRPC", "live")
            self._live_grpc_mode_container.setVisible(True)
            self._live_grpc_panel.update_live_grpc_mode_info()

        elif isinstance(self.parent.frame_source, RemoteHdf5Source):
            self._set_live_polling(False)
            self._set_mode_badge("Remote HDF5", "remote")
            self._remote_hdf5_mode_container.setVisible(True)
            self._update_remote_hdf5_mode_info()

        else:
            self._set_live_polling(False)
            label = getattr(self.parent.frame_source, "frame_source_label", "Unknown")
            self._set_mode_badge(str(label), "unknown")
            self._set_default_values()

    def _hide_all_mode_containers(self) -> None:
        """Hide all mode-specific containers."""
        self._file_mode_container.setVisible(False)
        self._live_grpc_mode_container.setVisible(False)
        self._remote_hdf5_mode_container.setVisible(False)

    def _set_mode_badge(self, text: str, mode: str) -> None:
        """Set mode badge text using the active application theme."""
        theme = current_theme()
        color = {
            "file": theme.accent,
            "live": theme.warning,
            "remote": theme.success,
        }.get(mode, theme.border_primary)
        text_color = self._badge_text_color(color)
        self.widgets["mode_badge"].setText(text)
        self.widgets["mode_badge"].setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                color: {text_color};
                padding: 2px 8px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }}
        """)

    @staticmethod
    def _badge_text_color(background_hex: str) -> str:
        """Choose black or white text for readable badge contrast."""
        color = QColor(background_hex)
        if not color.isValid():
            return "#ffffff"
        luminance = 0.2126 * color.redF() + 0.7152 * color.greenF() + 0.0722 * color.blueF()
        return "#111827" if luminance > 0.62 else "#ffffff"

    def _set_default_values(self) -> None:
        """Set default/empty values when no frame source available."""
        self.widgets["format_type"].setText(EMPTY_VALUE)
        self.widgets["file_count"].setText(EMPTY_VALUE)
        self.widgets["file_path"].setText(EMPTY_VALUE)
        self.widgets["chunk_info"].setText(EMPTY_VALUE)
        self.widgets["load_time"].setText(EMPTY_VALUE)
        self.widgets["preload_status"].setText(EMPTY_VALUE)
        self.widgets["loading_mode"].setText(EMPTY_VALUE)
        self.widgets["memory_usage"].setText(EMPTY_VALUE)
        self.widgets["total_frames"].setText(EMPTY_VALUE)
        self.widgets["loaded_frames"].setText(EMPTY_VALUE)
        self.widgets["frame_range"].setText(EMPTY_VALUE)
        self.widgets["file_size"].setText(EMPTY_VALUE)

    # File Mode Update

    def _update_file_mode_info(self) -> None:
        """Update local HDF5 information, opening the provider if necessary."""
        try:
            frame_source = self.parent.frame_source
            if not hasattr(frame_source, "provider") or frame_source.provider is None:
                frame_source.open()

            provider = frame_source.provider

            if not isinstance(provider, Hdf5Provider):
                self._set_default_values()
                return

            if provider.is_bulk:
                format_type = "HDF5 (Bulk)"
                if len(provider.bulk_files) > 1:
                    format_type += f" - {len(provider.bulk_files)} chunks"
            else:
                format_type = "HDF5 (Per-Frame)"
            self.widgets["format_type"].setText(format_type)

            if provider.is_bulk:
                file_count = len(provider.bulk_files)
            else:
                frames = provider.list_frames()
                file_count = len(frames)
            self.widgets["file_count"].setText(str(file_count))

            file_path = str(frame_source.root)
            self.widgets["file_path"].setText(file_path)

            if provider.is_bulk:
                if len(provider.bulk_files) > 1:
                    import os

                    chunk_sizes = []
                    for chunk_file in provider.bulk_files:
                        if os.path.exists(chunk_file):
                            size_mb = os.path.getsize(chunk_file) / (1024 * 1024)
                            chunk_sizes.append(f"{size_mb:.1f} MB")
                    chunk_info = (
                        f"{len(provider.bulk_files)} chunks, " f"{', '.join(chunk_sizes[:3])}"
                    )
                    if len(chunk_sizes) > 3:
                        chunk_info += "..."
                    self.widgets["chunk_info"].setText(chunk_info)
                else:
                    self.widgets["chunk_info"].setText("Single file")
            else:
                self.widgets["chunk_info"].setText("N/A")

            self._update_loading_metrics(provider)
            self._update_frame_statistics(provider)
            self._update_file_size(provider, frame_source)

        except (OSError, AttributeError, KeyError, ValueError) as e:
            logger.error("Error updating file mode info: %s", e)
            self._set_default_values()

    def _update_loading_metrics(self, provider: Any) -> None:
        """Update preload/lazy-load labels from provider and animation service."""
        anim_service = getattr(self.parent, "animation_service", None)
        total_frames = len(provider.list_frames()) if provider else 0

        if getattr(self.parent, "use_preload_mode", False) and anim_service is not None:
            duration = anim_service.preload_duration()
            if anim_service.preloading_started:
                if anim_service.preloading_completed:
                    if duration is not None:
                        self.widgets["load_time"].setText(
                            f"{duration:.2f} s" if duration >= 1.0 else f"{duration*1000:.0f} ms"
                        )
                    else:
                        self.widgets["load_time"].setText("Completed")
                else:
                    text = "Loading..."
                    if duration is not None:
                        text = (
                            f"{duration:.1f} s (loading...)"
                            if duration >= 1.0
                            else f"{duration*1000:.0f} ms (loading...)"
                        )
                    self.widgets["load_time"].setText(text)
            else:
                self.widgets["load_time"].setText("Not started")
        else:
            self.widgets["load_time"].setText("On-demand")

        if getattr(self.parent, "use_preload_mode", False) and anim_service is not None:
            loaded_count = anim_service.preload_frame_count
            if total_frames > 0:
                progress = int(loaded_count / total_frames * 100)
            else:
                progress = 0
            if anim_service.preloading_started:
                if anim_service.preloading_completed:
                    self.widgets["preload_status"].setText(
                        f"Complete ({loaded_count}/{total_frames})"
                    )
                else:
                    self.widgets["preload_status"].setText(
                        f"In Progress ({loaded_count}/{total_frames}, {progress}%)"
                    )
            else:
                self.widgets["preload_status"].setText("Enabled (Not Started)")
        else:
            self.widgets["preload_status"].setText(EMPTY_VALUE)

        if hasattr(self.parent, "use_preload_mode"):
            if self.parent.use_preload_mode:
                self.widgets["loading_mode"].setText("Preload")
            else:
                self.widgets["loading_mode"].setText("Lazy Load")
        else:
            self.widgets["loading_mode"].setText(EMPTY_VALUE)

        # Memory usage (estimated)
        if anim_service is not None and anim_service.preload_frame_count > 0:
            estimated_mb = anim_service.preload_frame_count * 1.0
            if estimated_mb < 1024:
                self.widgets["memory_usage"].setText(f"~{estimated_mb:.1f} MB")
            else:
                self.widgets["memory_usage"].setText(f"~{estimated_mb/1024:.2f} GB")
        else:
            self.widgets["memory_usage"].setText("Minimal")

    def _update_frame_statistics(self, provider: Any) -> None:
        """Update frame count, loaded count, and frame-index range labels."""
        try:
            frames = provider.list_frames()
            total_frames = len(frames)
            self.widgets["total_frames"].setText(str(total_frames))

            anim_service = getattr(self.parent, "animation_service", None)
            if anim_service is not None:
                loaded_count = anim_service.preload_frame_count
                self.widgets["loaded_frames"].setText(f"{loaded_count}/{total_frames}")
            else:
                self.widgets["loaded_frames"].setText("0")

            if frames:
                frame_range = f"{min(frames)} - {max(frames)}"
                self.widgets["frame_range"].setText(frame_range)
            else:
                self.widgets["frame_range"].setText(EMPTY_VALUE)
        except (OSError, KeyError, ValueError) as e:
            logger.error("Error updating frame statistics: %s", e)
            self.widgets["total_frames"].setText(EMPTY_VALUE)
            self.widgets["loaded_frames"].setText(EMPTY_VALUE)
            self.widgets["frame_range"].setText(EMPTY_VALUE)

    def _update_file_size(self, provider: Any, frame_source: Any) -> None:
        """Estimate on-disk HDF5 size for bulk or per-frame layouts."""
        try:
            import os

            total_size = 0

            if provider.is_bulk:
                for bulk_file in provider.bulk_files:
                    if os.path.exists(bulk_file):
                        total_size += os.path.getsize(bulk_file)
            else:
                frames_dir = os.path.join(str(frame_source.root), "frames")
                if os.path.exists(frames_dir):
                    import glob

                    hdf5_files = glob.glob(os.path.join(frames_dir, "mpc_frames_*.h5"))
                    for hdf5_file in hdf5_files:
                        total_size += os.path.getsize(hdf5_file)

            if total_size < 1024:
                size_str = f"{total_size} B"
            elif total_size < 1024 * 1024:
                size_str = f"{total_size / 1024:.2f} KB"
            elif total_size < 1024 * 1024 * 1024:
                size_str = f"{total_size / (1024 * 1024):.2f} MB"
            else:
                size_str = f"{total_size / (1024 * 1024 * 1024):.2f} GB"

            self.widgets["file_size"].setText(size_str)
        except OSError as e:
            logger.error("Error calculating file size: %s", e)
            self.widgets["file_size"].setText(EMPTY_VALUE)

    # Remote HDF5 Mode Update

    def _update_remote_hdf5_mode_info(self) -> None:
        """Update remote-HDF5 labels without opening the remote connection."""
        try:
            frame_source = self.parent.frame_source
            self.widgets["remote_server_address"].setText(frame_source.server_address)

            provider = getattr(frame_source, "provider", None)
            if provider is not None and hasattr(provider, "is_connected"):
                if provider.is_connected:
                    self.widgets["remote_connection_status"].setText("Connected")
                    set_widget_role(self.widgets["remote_connection_status"], "success")
                else:
                    self.widgets["remote_connection_status"].setText("Disconnected")
                    set_widget_role(self.widgets["remote_connection_status"], "error")
            else:
                self.widgets["remote_connection_status"].setText("Not connected")
                set_widget_role(self.widgets["remote_connection_status"], None)

            metadata = frame_source.metadata if hasattr(frame_source, "metadata") else {}
            scene_name = metadata.get("scene_name", metadata.get("scenario_name", EMPTY_VALUE))
            self.widgets["remote_scene_name"].setText(str(scene_name))
            frame_set_id = str(metadata.get("frame_set_id") or "")
            self.widgets["remote_frame_set_id"].setText(
                frame_set_id[:12] if frame_set_id else EMPTY_VALUE
            )
            snapshot_valid = metadata.get("snapshot_valid")
            snapshot_error = metadata.get("snapshot_error") or ""
            if snapshot_valid is True:
                self.widgets["remote_snapshot_status"].setText("Valid")
                set_widget_role(self.widgets["remote_snapshot_status"], "success")
            elif snapshot_valid is False:
                self.widgets["remote_snapshot_status"].setText(snapshot_error or "Changed")
                set_widget_role(self.widgets["remote_snapshot_status"], "error")
            else:
                self.widgets["remote_snapshot_status"].setText(EMPTY_VALUE)
                set_widget_role(self.widgets["remote_snapshot_status"], None)

            self.widgets["remote_cache_size"].setText(f"{frame_source.cache_size} frames")

            cached_count = 0
            if provider is not None and hasattr(provider, "cached_frame_count"):
                cached_count = provider.cached_frame_count
            self.widgets["remote_cached_frames"].setText(str(cached_count))

            utilization = (
                (cached_count / frame_source.cache_size * 100) if frame_source.cache_size > 0 else 0
            )
            self.widgets["remote_cache_utilization"].setText(f"{utilization:.1f}%")

            if provider is not None and hasattr(provider, "cache_stats"):
                stats = provider.cache_stats
                total_requests = stats.get("total", 0)
                if total_requests > 0:
                    ratio = stats.get("hit_ratio", 0.0)
                    hits = stats.get("hits", 0)
                    misses = stats.get("misses", 0)
                    self.widgets["remote_cache_hit_ratio"].setText(
                        f"{ratio:.0%} ({hits} hits / {misses} misses)"
                    )
                else:
                    self.widgets["remote_cache_hit_ratio"].setText(EMPTY_VALUE)
                latency = stats.get("last_fetch_latency_ms", 0.0)
                if latency > 0:
                    self.widgets["remote_fetch_latency"].setText(f"{latency:.1f} ms")
                else:
                    self.widgets["remote_fetch_latency"].setText(EMPTY_VALUE)
            else:
                self.widgets["remote_cache_hit_ratio"].setText(EMPTY_VALUE)
                self.widgets["remote_fetch_latency"].setText(EMPTY_VALUE)

            if provider is not None:
                frames = provider.list_frames()
                total_frames = len(frames)
                self.widgets["remote_total_frames"].setText(str(total_frames))

                if frames:
                    frame_range = f"{min(frames)} - {max(frames)}"
                    self.widgets["remote_frame_range"].setText(frame_range)
                else:
                    self.widgets["remote_frame_range"].setText(EMPTY_VALUE)
            else:
                self.widgets["remote_total_frames"].setText(EMPTY_VALUE)
                self.widgets["remote_frame_range"].setText(EMPTY_VALUE)

        except (OSError, AttributeError, KeyError, ValueError) as e:
            logger.error("Error updating remote HDF5 mode info: %s", e)
            self._set_remote_default_values()

    def _set_remote_default_values(self) -> None:
        """Reset remote-HDF5 status widgets to their disconnected defaults."""
        self.widgets["remote_server_address"].setText(EMPTY_VALUE)
        self.widgets["remote_connection_status"].setText(EMPTY_VALUE)
        set_widget_role(self.widgets["remote_connection_status"], None)
        self.widgets["remote_scene_name"].setText(EMPTY_VALUE)
        self.widgets["remote_frame_set_id"].setText(EMPTY_VALUE)
        self.widgets["remote_snapshot_status"].setText(EMPTY_VALUE)
        set_widget_role(self.widgets["remote_snapshot_status"], None)
        self.widgets["remote_cache_size"].setText(EMPTY_VALUE)
        self.widgets["remote_cached_frames"].setText(EMPTY_VALUE)
        self.widgets["remote_cache_utilization"].setText(EMPTY_VALUE)
        self.widgets["remote_cache_hit_ratio"].setText(EMPTY_VALUE)
        self.widgets["remote_fetch_latency"].setText(EMPTY_VALUE)
        self.widgets["remote_total_frames"].setText(EMPTY_VALUE)
        self.widgets["remote_frame_range"].setText(EMPTY_VALUE)
