"""Runtime diagnostics and cache telemetry widgets for the System tab.

``PerformancePanel`` displays renderer frame stats, frame/cache provenance, and
preload state. It deliberately polls only while its collapsible section is
visible in its active workflow tab so simply constructing the visualizer does
not add idle renderer overhead.
"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from shared.logging import get_current_log_level_name

from ..playback import format_playback_fps
from .base import BasePanel

REFRESH_INTERVAL_MS = 1000
LOG_LEVEL_OPTIONS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


class PerformancePanel(BasePanel):
    """Create performance widgets and refresh them only when visible."""

    def __init__(self, parent_widget: Any) -> None:
        """Initialize lazy refresh state without starting a timer."""
        super().__init__(parent_widget)
        self._section = None
        self._tab_widget = None
        self._refresh_timer: Optional[QTimer] = None
        self._deferred_sync_timer: Optional[QTimer] = None
        self._refresh_cache_snapshot: Optional[dict[str, Any]] = None

    def create_panel(self) -> QGroupBox:
        """Build grouped runtime, cache, preload, and diagnostics controls."""
        group = self.create_group_box("Performance")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._create_runtime_group())
        layout.addWidget(self._create_cache_group())
        layout.addWidget(self._create_preload_status_group())
        layout.addWidget(self._create_diagnostics_group())
        return group

    def bind_section(self, section: Any) -> None:
        """Bind this panel to its collapsible section for refresh gating."""
        self._section = section
        if hasattr(section, "toggled"):
            section.toggled.connect(self._on_section_toggled)
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(self._sync_refresh_timer)
        timer.start(0)
        self._deferred_sync_timer = timer

    def cleanup(self) -> None:
        """Stop polling and cancel callbacks queued during panel construction."""
        deferred = self._deferred_sync_timer
        self._deferred_sync_timer = None
        if deferred is not None:
            deferred.stop()
            deferred.deleteLater()

        timer = self._refresh_timer
        self._refresh_timer = None
        if timer is not None:
            timer.stop()
            timer.deleteLater()

        section = self._section
        if section is not None and hasattr(section, "toggled"):
            try:
                section.toggled.disconnect(self._on_section_toggled)
            except (RuntimeError, TypeError):
                pass
        tab_widget = self._tab_widget
        if tab_widget is not None:
            try:
                tab_widget.currentChanged.disconnect(self._on_tab_changed)
            except (RuntimeError, TypeError):
                pass
        self._section = None
        self._tab_widget = None

    def _create_runtime_group(self) -> QGroupBox:
        """Create renderer/frame-time labels updated from runtime stats."""
        group = self.create_subgroup_box("Runtime")
        layout = QGridLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 4, 6, 6)

        for row, (label, key) in enumerate(
            [
                ("Playback updates:", "perf_playback_updates_value"),
                ("Update work:", "perf_frame_ms_value"),
                ("Work average:", "perf_avg_frame_ms_value"),
                ("Update interval p95:", "perf_update_p95_value"),
                ("Render callback:", "perf_present_value"),
                ("Renderer submit:", "perf_draw_value"),
                ("Render jitter:", "perf_jitter_value"),
            ]
        ):
            self._add_metric_row(layout, row, label, key)
        return group

    def _create_cache_group(self) -> QGroupBox:
        """Create cache provenance labels for frame and derived payload caches."""
        group = self.create_subgroup_box("Cache")
        layout = QGridLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 4, 6, 6)

        for row, (label, key) in enumerate(
            [
                ("Source:", "perf_frame_source_value"),
                ("Preloaded:", "perf_preloaded_value"),
                ("Frames:", "perf_frame_cache_value"),
                ("ViewModel:", "perf_viewmodel_cache_value"),
                ("Coverage:", "perf_coverage_cache_value"),
                ("Targets:", "perf_target_cache_value"),
                ("Assets:", "perf_asset_cache_value"),
            ]
        ):
            self._add_metric_row(layout, row, label, key)
        return group

    def _create_preload_status_group(self) -> QGroupBox:
        """Create preload status and cache-action controls."""
        group = self.create_subgroup_box("Preload Status")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 4, 6, 6)

        self.widgets["preload_status_label"] = QLabel("Preload: Not started")
        self.widgets["preload_status_label"].setStyleSheet("font-size: 11px;")
        layout.addWidget(self.widgets["preload_status_label"])

        row = QHBoxLayout()
        row.setSpacing(6)
        self.widgets["restart_preload_btn"] = QPushButton("Restart Preload")
        self.widgets["restart_preload_btn"].setToolTip("Restart background frame preloading")
        row.addWidget(self.widgets["restart_preload_btn"])
        self.widgets["clear_cache_btn"] = QPushButton("Clear Frames")
        self.widgets["clear_cache_btn"].setToolTip(
            "Clear transient frames, derived ViewModels, and renderer frame state"
        )
        row.addWidget(self.widgets["clear_cache_btn"])
        self.widgets["clear_asset_cache_btn"] = QPushButton("Clear Assets")
        self.widgets["clear_asset_cache_btn"].setToolTip(
            "Explicitly clear inactive target assets, decoded/native textures, "
            "prepared meshes, generated UVs, and reusable scene payloads. "
            "The next load may be slower."
        )
        row.addWidget(self.widgets["clear_asset_cache_btn"])
        row.addStretch()
        layout.addLayout(row)
        return group

    def _create_diagnostics_group(self) -> QGroupBox:
        """Create app-wide diagnostic controls exposed through UIController."""
        group = self.create_subgroup_box("Diagnostics")
        layout = QHBoxLayout(group)
        layout.setSpacing(8)
        layout.setContentsMargins(6, 4, 6, 6)

        layout.addWidget(QLabel("Log Level:"))
        self.widgets["log_level_combo"] = QComboBox()
        self.widgets["log_level_combo"].addItems(LOG_LEVEL_OPTIONS)
        self.widgets["log_level_combo"].setCurrentText(get_current_log_level_name())
        self.widgets["log_level_combo"].setToolTip(
            "Set logging verbosity (WARNING=minimal, INFO=normal, DEBUG=detailed; "
            "ERROR and CRITICAL show failures only)"
        )
        layout.addWidget(self.widgets["log_level_combo"])
        layout.addStretch()
        return group

    def _add_metric_row(self, layout: QGridLayout, row: int, label: str, key: str) -> None:
        """Register one metric value label under a stable widget key."""
        name = QLabel(label)
        name.setStyleSheet("font-size: 10px;")
        layout.addWidget(name, row, 0)
        value = QLabel("--")
        value.setStyleSheet("font-size: 10px;")
        self.widgets[key] = value
        layout.addWidget(value, row, 1)

    def _on_section_toggled(self, _expanded: bool) -> None:
        """Re-evaluate polling whenever the collapsible section changes."""
        self._sync_refresh_timer()

    def _refresh_timer_obj(self) -> QTimer:
        """Create the lazy refresh timer on first use."""
        if self._refresh_timer is None:
            self._refresh_timer = QTimer()
            self._refresh_timer.setInterval(REFRESH_INTERVAL_MS)
            self._refresh_timer.timeout.connect(self.refresh_metrics)
        return self._refresh_timer

    def _bind_tab_signal(self) -> None:
        """Subscribe once to active-tab changes so polling follows visibility."""
        if self._tab_widget is not None:
            return
        manager = getattr(self.parent, "ui_manager", None)
        tab_widget = getattr(manager, "_tab_widget", None)
        if tab_widget is None:
            return
        tab_widget.currentChanged.connect(self._on_tab_changed)
        self._tab_widget = tab_widget

    def _on_tab_changed(self, _index: int) -> None:
        """Re-evaluate polling when the active workflow tab changes."""
        self._sync_refresh_timer()

    def _section_active(self) -> bool:
        """Return whether metrics are visible enough to justify polling."""
        section = self._section
        if section is None:
            return False
        try:
            if hasattr(section, "is_expanded") and not section.is_expanded():
                return False
            if hasattr(section, "isVisible") and not section.isVisible():
                return False
        except RuntimeError:
            return False
        manager = getattr(self.parent, "ui_manager", None)
        if manager is None:
            return True
        is_panel_active = getattr(manager, "is_panel_in_active_tab", None)
        if not callable(is_panel_active):
            return True
        return bool(is_panel_active("performance"))

    def _sync_refresh_timer(self) -> None:
        """Start or stop polling according to section and active-tab state."""
        self._bind_tab_signal()
        timer = self._refresh_timer_obj()
        if self._section_active():
            self.refresh_metrics()
            if not timer.isActive():
                timer.start()
        elif timer.isActive():
            timer.stop()

    def refresh_metrics(self) -> None:
        """Refresh diagnostics only when the section is expanded and visible."""
        if not self._section_active():
            timer = self._refresh_timer
            if timer is not None and timer.isActive():
                timer.stop()
            return

        self._sync_source_actions()
        stats = self._renderer_stats()
        self._refresh_cache_snapshot = self._cache_telemetry(
            getattr(self.parent, "cache_service", None)
        )
        self._set("perf_playback_updates_value", self._format_playback_updates())
        self._set(
            "perf_frame_ms_value",
            self._format_ms(getattr(self.parent, "last_frame_duration_ms", None)),
        )
        self._set("perf_avg_frame_ms_value", self._format_avg_frame_ms())
        self._set("perf_update_p95_value", self._format_update_interval_p95())
        self._set("perf_present_value", self._format_present(stats))
        self._set("perf_draw_value", self._format_ms(stats.get("avg_draw_ms")))
        self._set("perf_jitter_value", self._format_jitter(stats))
        self._set("perf_frame_source_value", self._format_frame_source())
        self._set("perf_preloaded_value", self._format_preloaded())
        self._set("perf_frame_cache_value", self._format_frame_cache())
        self._set("perf_viewmodel_cache_value", self._format_viewmodel_cache())
        self._set("perf_coverage_cache_value", self._format_coverage_cache())
        self._set("perf_target_cache_value", self._format_target_cache())
        self._set("perf_asset_cache_value", self._format_asset_caches())
        self._set("preload_status_label", self._format_preload_status())
        self._refresh_cache_snapshot = None

    def _sync_source_actions(self) -> None:
        """Hide preload actions that do not apply to live streaming."""
        from ..io.frame_sources import LiveGrpcSource

        frame_source = getattr(self.parent, "frame_source", None)
        restart = self.widgets.get("restart_preload_btn")
        if restart is not None:
            restart.setVisible(not isinstance(frame_source, LiveGrpcSource))

    def _set(self, key: str, value: str) -> None:
        """Set a registered label when it still exists."""
        widget = self.widgets.get(key)
        if widget is not None and hasattr(widget, "setText"):
            widget.setText(value)

    def _renderer_stats(self) -> dict[str, Any]:
        """Fetch renderer stats defensively through the optional protocol hook."""
        renderer = getattr(self.parent, "renderer", None)
        get_stats = getattr(renderer, "get_runtime_stats", None)
        if not callable(get_stats):
            return {}
        try:
            stats = get_stats() or {}
        except (RuntimeError, ValueError, TypeError, AttributeError):
            return {}
        return stats if isinstance(stats, dict) else {}

    def _format_playback_updates(self) -> str:
        """Format backend-neutral completed scenario-frame update cadence."""
        tracker = getattr(self.parent, "playback_cadence", None)
        value = tracker.frames_per_second() if tracker is not None else None
        value_text = format_playback_fps(value)
        if value_text is None:
            return "measuring" if getattr(self.parent, "animation_running", False) else "--"
        formatted = f"{value_text} updates/s"
        return (
            formatted if getattr(self.parent, "animation_running", False) else f"{formatted} (last)"
        )

    def _format_update_interval_p95(self) -> str:
        """Format recent p95 spacing between completed playback updates."""
        tracker = getattr(self.parent, "playback_cadence", None)
        percentile = getattr(tracker, "percentile_interval_ms", None)
        if not callable(percentile):
            return "--"
        return self._format_ms(percentile(95.0))

    def _format_avg_frame_ms(self) -> str:
        """Format recent app frame duration using the last 30 frame samples."""
        frame_times = getattr(self.parent, "frame_times", None) or []
        if not frame_times:
            return "--"
        window = frame_times[-min(len(frame_times), 30) :]
        avg_ms = (sum(window) / len(window)) * 1000.0
        return self._format_ms(avg_ms)

    def _format_present(self, stats: dict[str, Any]) -> str:
        """Combine renderer callback cadence and update-to-callback latency."""
        utp_ms = self._format_ms(stats.get("avg_update_to_present_ms"))
        present_fps = self._format_number(
            stats.get("recent_present_fps", stats.get("effective_present_fps")),
            suffix=" FPS",
            decimals=0,
        )
        if utp_ms == "--" and present_fps == "--":
            return "--"
        if utp_ms == "--":
            return present_fps
        if present_fps == "--":
            return utp_ms
        return f"{present_fps}, {utp_ms} latency"

    def _format_jitter(self, stats: dict[str, Any]) -> str:
        """Combine present jitter and frame drop percentage."""
        jitter = self._format_ms(stats.get("present_jitter_ms"))
        drop_rate = stats.get("frame_drop_rate")
        drop_text = self._format_number(
            None if drop_rate is None else float(drop_rate) * 100.0,
            suffix="%",
            decimals=1,
        )
        if jitter == "--" and drop_text == "--":
            return "--"
        if jitter == "--":
            return drop_text
        if drop_text == "--":
            return jitter
        return f"{jitter}, {drop_text}"

    def _format_frame_source(self) -> str:
        """Describe the active frame's source within local cache layers."""
        step = getattr(self.parent, "animation_step", None)
        cache = getattr(self.parent, "cache_service", None)
        if step is None or cache is None:
            return "--"
        try:
            if cache.is_override(step):
                return "override"
            if cache.is_preloaded(step):
                return "preloaded"
            if cache.has_frame(step):
                return "cached"
        except (RuntimeError, TypeError, ValueError):
            return "--"
        return "on-demand"

    def _format_preloaded(self) -> str:
        """Format preloaded frame count against total known animation steps."""
        anim = getattr(self.parent, "animation_service", None)
        if anim is None:
            return "--"
        count = getattr(anim, "preload_frame_count", 0)
        total = int(getattr(self.parent, "total_animation_steps", 0) or 0)
        return f"{count}/{total}" if total > 0 else str(count)

    def _format_frame_cache(self) -> str:
        """Format raw-frame cache size and optional hit/miss/eviction counters."""
        cache_service = getattr(self.parent, "cache_service", None)
        telemetry = self._current_cache_telemetry(cache_service)
        if telemetry:
            size = telemetry.get("frame_cache_size")
            max_size = telemetry.get("frame_cache_max_size")
            if size is None:
                return "--"
            text = f"{size}/{max_size}" if max_size is not None else str(size)
            hits = telemetry.get("frame_cache_hits")
            misses = telemetry.get("frame_cache_misses")
            evictions = telemetry.get("frame_cache_evictions")
            if hits is not None and misses is not None and evictions is not None:
                text += f" h{hits}/m{misses}/e{evictions}"
            return text

        cache = getattr(cache_service, "frame_cache", None)
        if cache is None:
            return "--"
        size = getattr(cache, "size", None)
        max_size = getattr(cache, "max_size", None)
        if size is None:
            return "--"
        return f"{size}/{max_size}" if max_size is not None else str(size)

    def _format_viewmodel_cache(self) -> str:
        """Format ViewModel cache and renderer MPC line-cache telemetry."""
        telemetry = self._current_cache_telemetry(getattr(self.parent, "cache_service", None))
        if telemetry:
            size = telemetry.get("view_model_cache_size")
            text = "--" if size is None else str(size)
            line_bytes = telemetry.get("mpc_line_cache_bytes")
            line_max = telemetry.get("mpc_line_cache_max_bytes")
            if line_bytes is not None and line_max:
                text += (
                    " line "
                    f"{self._format_bytes_mb(line_bytes)}/{self._format_bytes_mb(line_max)}"
                )
                hits = telemetry.get("mpc_line_cache_hits")
                misses = telemetry.get("mpc_line_cache_misses")
                evictions = telemetry.get("mpc_line_cache_evictions")
                if hits is not None and misses is not None and evictions is not None:
                    text += f" h{hits}/m{misses}/e{evictions}"
            return text

        cache = getattr(self.parent, "mpc_view_cache", None)
        try:
            return str(len(cache)) if cache is not None else "--"
        except TypeError:
            return "--"

    def _format_target_cache(self) -> str:
        """Format bounded target-frame assets and lookahead activity."""
        telemetry = self._current_cache_telemetry(getattr(self.parent, "cache_service", None))
        entries = telemetry.get("target_asset_cache_entries")
        if entries is None:
            return "--"
        max_entries = telemetry.get("target_asset_cache_max_entries")
        text = f"{entries}/{max_entries}" if max_entries is not None else str(entries)
        byte_count = telemetry.get("target_asset_cache_bytes")
        max_bytes = telemetry.get("target_asset_cache_max_bytes")
        if byte_count is not None:
            text += f", {self._format_bytes_mb(byte_count)}"
            if max_bytes:
                text += f"/{self._format_bytes_mb(max_bytes)}"
        hits = telemetry.get("target_asset_cache_hits")
        misses = telemetry.get("target_asset_cache_misses")
        evictions = telemetry.get("target_asset_cache_evictions")
        if hits is not None and misses is not None and evictions is not None:
            text += f" h{hits}/m{misses}/e{evictions}"
        pending = telemetry.get("target_asset_cache_pending")
        if pending:
            text += f" p{pending}"
        return text

    def _format_asset_caches(self) -> str:
        """Format aggregate reusable CPU, disk, and native asset ownership."""
        telemetry = self._current_cache_telemetry(getattr(self.parent, "cache_service", None))
        aggregate = telemetry.get("static_asset_cache_aggregate")
        if not isinstance(aggregate, dict):
            return "--"

        parts: list[str] = []
        for key, label in (("memory", "RAM"), ("disk", "disk")):
            bucket = aggregate.get(key)
            if not isinstance(bucket, dict):
                continue
            byte_count = bucket.get("bytes")
            max_bytes = bucket.get("max_bytes")
            if byte_count is None:
                continue
            formatted = f"{label} {self._format_bytes_mb(byte_count)}"
            if max_bytes:
                formatted += f"/{self._format_bytes_mb(max_bytes)}"
            parts.append(formatted)

        native = aggregate.get("native")
        if isinstance(native, dict):
            native_entries = native.get("entries")
            if native_entries is not None:
                parts.append(f"native {native_entries}")
        return " | ".join(parts) if parts else "--"

    def _current_cache_telemetry(self, cache_service: Any) -> dict[str, Any]:
        """Reuse the one inventory snapshot captured for this refresh pass."""
        if self._refresh_cache_snapshot is not None:
            return self._refresh_cache_snapshot
        return self._cache_telemetry(cache_service)

    @staticmethod
    def _cache_telemetry(cache_service: Any) -> dict[str, Any]:
        """Return cache-service telemetry only when the optional hook is valid."""
        get_telemetry = getattr(cache_service, "get_cache_telemetry", None)
        if not callable(get_telemetry):
            return {}
        try:
            telemetry = get_telemetry() or {}
        except (RuntimeError, ValueError, TypeError, AttributeError):
            return {}
        return telemetry if isinstance(telemetry, dict) else {}

    @staticmethod
    def _format_bytes_mb(value: Any) -> str:
        """Format byte counts in megabytes for compact cache labels."""
        try:
            mb = float(value) / (1024.0 * 1024.0)
        except (TypeError, ValueError):
            return "--"
        if mb >= 10.0:
            return f"{mb:.0f} MB"
        return f"{mb:.1f} MB"

    def _format_coverage_cache(self) -> str:
        """Format coverage mesh cache size from the coverage service."""
        service = getattr(self.parent, "coverage_service", None)
        stats_fn = getattr(service, "stats", None)
        if not callable(stats_fn):
            return "--"
        try:
            stats = stats_fn() or {}
        except (RuntimeError, ValueError, TypeError, AttributeError):
            return "--"
        size = stats.get("cache_size")
        max_size = stats.get("max_cache_size")
        if size is None:
            return "--"
        return f"{size}/{max_size}" if max_size is not None else str(size)

    def _format_preload_status(self) -> str:
        """Format preload status, excluding live gRPC streams from preload work."""
        anim = getattr(self.parent, "animation_service", None)
        frame_source = getattr(self.parent, "frame_source", None)
        source_name = type(frame_source).__name__ if frame_source is not None else ""
        if anim is None:
            return "Preload: Unavailable"
        if source_name == "LiveGrpcSource":
            return "Preload: Not needed (live stream)"
        if not bool(getattr(self.parent, "use_preload_mode", True)):
            return "Preload: On-demand mode"

        count = getattr(anim, "preload_frame_count", 0)
        total = int(getattr(self.parent, "total_animation_steps", 0) or 0)
        suffix = f" ({count}/{total})" if total > 0 else f" ({count})"

        if getattr(anim, "preloading_completed", False):
            return "Preload: Complete" + suffix
        if getattr(anim, "preloading_started", False):
            duration_attr = getattr(anim, "preload_duration", None)
            duration = duration_attr() if callable(duration_attr) else duration_attr
            elapsed = "" if duration is None else f", {duration:.1f}s"
            return "Preload: Loading" + suffix + elapsed
        return "Preload: Not started"

    @staticmethod
    def _format_ms(value: Any) -> str:
        """Format one numeric value as milliseconds."""
        return PerformancePanel._format_number(value, suffix=" ms", decimals=1)

    @staticmethod
    def _format_number(value: Any, *, suffix: str = "", decimals: int = 1) -> str:
        """Format numbers for labels, using ``--`` for missing or invalid data."""
        if value is None:
            return "--"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "--"
        if decimals <= 0:
            return f"{number:.0f}{suffix}"
        return f"{number:.{decimals}f}{suffix}"
