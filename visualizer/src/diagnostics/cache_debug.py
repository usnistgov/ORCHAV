"""Opt-in cache and host-memory inspection without reading simulation payloads.

Sampling uses existing counters and retained-byte accounting. Disk-cache sizes
are last-published inventories, not directory scans; cache sizes may share array
ownership and must not be summed to estimate process memory.
"""

from __future__ import annotations

import ctypes
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PySide6.QtCore import QEvent, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .cache_telemetry import cache_telemetry_snapshot


def _linux_memory() -> dict[str, int]:
    """Read small kernel-provided memory summaries, never process allocations."""

    def values(path: str) -> dict[str, int]:
        result = {}
        for line in Path(path).read_text().splitlines():
            name, _, value = line.partition(":")
            fields = value.split()
            if len(fields) == 2 and fields[1] == "kB":
                result[name] = int(fields[0]) * 1024
        return result

    process = values("/proc/self/status")
    system = values("/proc/meminfo")
    return {
        "process_rss_bytes": process["VmRSS"],
        "process_swap_bytes": process.get("VmSwap", 0),
        "system_total_bytes": system["MemTotal"],
        "system_available_bytes": system["MemAvailable"],
        "system_swap_used_bytes": system["SwapTotal"] - system["SwapFree"],
        "system_swap_total_bytes": system["SwapTotal"],
    }


def _windows_memory() -> dict[str, int]:
    """Read Windows working-set/private-commit and physical-memory counters."""

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
            (name, ctypes.c_uint64)
            for name in (
                "total",
                "available",
                "commit_limit",
                "commit_available",
                "virtual",
                "virtual_available",
                "extended",
            )
        ]

    class ProcessMemory(ctypes.Structure):
        _fields_ = [("size", ctypes.c_uint32), ("faults", ctypes.c_uint32)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak_rss",
                "rss",
                "peak_paged_pool",
                "paged_pool",
                "peak_nonpaged_pool",
                "nonpaged_pool",
                "pagefile",
                "peak_pagefile",
                "private",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
    kernel.GlobalMemoryStatusEx.restype = ctypes.c_int
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemory),
        ctypes.c_uint32,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    system, process = MemoryStatus(), ProcessMemory()
    system.length, process.size = ctypes.sizeof(system), ctypes.sizeof(process)
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(system)) or not psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(process), ctypes.sizeof(process)
    ):
        raise OSError("Windows memory counters unavailable")
    return {
        "process_rss_bytes": process.rss,
        "process_private_commit_bytes": process.private,
        "system_total_bytes": system.total,
        "system_available_bytes": system.available,
        # Available commit is process headroom, not physical free RAM.
        "process_commit_headroom_bytes": system.commit_available,
    }


def memory_snapshot() -> dict[str, Any]:
    """Return current host/process memory, or a truthful unavailable notice."""
    try:
        if sys.platform == "win32":
            result = _windows_memory()
        elif sys.platform.startswith("linux"):
            result = _linux_memory()
        else:
            return {"unavailable": "Memory sampling is available on Windows and Linux"}
        result["system_used_bytes"] = (
            result["system_total_bytes"] - result["system_available_bytes"]
        )
        return result
    except (OSError, ValueError, KeyError, AttributeError) as exc:
        return {"unavailable": str(exc)}


def _call(owner: Any, name: str) -> dict:
    callback = getattr(owner, name, None)
    if not callable(callback):
        return {}
    try:
        return dict(callback() or {})
    except (OSError, RuntimeError, ValueError, TypeError, AttributeError):
        return {}


def _future_state(future: Any) -> str:
    if future is None:
        return "idle"
    if future.cancelled():
        return "cancelled"
    if future.done():
        return "complete, awaiting delivery"
    return "running" if future.running() else "queued"


def _length(value: Any) -> int | None:
    return len(value) if value is not None else None


def dashboard_session(visualizer: Any) -> Any:
    """Find the already-created dashboard without opening a panel or service."""
    panels = getattr(getattr(visualizer, "ui_manager", None), "panels", {})
    return getattr(panels.get("sensing"), "_dashboard_session", None)


def cache_debug_snapshot(visualizer: Any) -> dict[str, Any]:
    """Inspect known owners only; do not populate caches or enumerate disk files."""
    from ..config import MAX_VIEW_MODEL_CACHE_SIZE
    from ..sensing.dashboard_data import MAX_HISTORY_BYTES, MAX_HISTORY_RESULTS
    from ..sensing.recording_cache import MAX_DISK_BYTES

    rows = []

    def add(name: str, **values: Any) -> None:
        rows.append({"name": name, **values})

    frames = getattr(getattr(visualizer, "cache_service", None), "frame_cache", None)
    stats = getattr(frames, "stats", None)
    add(
        "Scene frames",
        storage="RAM",
        entries=getattr(frames, "size", None),
        max_entries=getattr(frames, "max_size", None),
        counters={name: getattr(stats, name, None) for name in ("hits", "misses", "evictions")},
        note="No byte budget; retained arrays can also belong to canonical MPC data",
    )
    core = getattr(visualizer, "mpc_core", None)
    add(
        "Canonical MPC data",
        storage="RAM",
        entries=_length(getattr(core, "_canon_cache", None)),
        bytes=getattr(core, "_canon_bytes_used", None),
        max_bytes=getattr(core, "_canon_max_bytes", None),
    )
    add(
        "ViewModels",
        storage="RAM",
        entries=_length(getattr(visualizer, "mpc_view_cache", None)),
        max_entries=MAX_VIEW_MODEL_CACHE_SIZE,
    )
    renderer = _call(getattr(visualizer, "renderer", None), "get_runtime_stats")
    add(
        "Expanded MPC lines",
        storage="RAM",
        **{key: renderer.get("mpc_line_cache_" + key) for key in ("entries", "bytes", "max_bytes")},
        counters={
            key: renderer.get("mpc_line_cache_" + key) for key in ("hits", "misses", "evictions")
        },
    )
    add(
        "Target assets",
        storage="RAM",
        **_call(getattr(visualizer, "target_asset_cache", None), "telemetry"),
    )
    coverage = _call(getattr(visualizer, "coverage_service", None), "stats")
    add(
        "Coverage",
        storage="RAM",
        entries=coverage.get("cache_size"),
        max_entries=coverage.get("max_cache_size"),
        counters={key: coverage.get(key) for key in ("hits", "misses", "evictions")},
    )

    assets = cache_telemetry_snapshot()
    for name, item in assets.items():
        if name == "target_assets":
            continue  # Its owner above also provides the limits and pending jobs.
        labels = {
            "texture_decode": ("Decoded textures", "RAM"),
            "pygfx_native_texture": ("Shared native textures", "native"),
            "pygfx_mesh_buffers.memory": ("Prepared meshes", "RAM"),
            "pygfx_mesh_buffers.disk": ("Prepared mesh files", "disk"),
            "scene_payload": ("Scene payload files", "disk"),
            "uv": ("Generated UVs", "RAM/disk"),
        }
        label, storage = labels.get(name, (name.replace("_", " "), "reported"))
        add(
            label,
            storage=storage,
            entries=item["current_entries"],
            bytes=item["current_bytes"],
            counters=item["counters"],
            elapsed_ms=item["elapsed_ms"],
            root=item["root"],
            note="Last published inventory; disk files are not rescanned",
        )

    session = dashboard_session(visualizer)
    sensing = {}
    if session is not None:
        service = getattr(session, "service", None)
        disk = getattr(service, "recording_cache", None)
        diagnostics = _call(session, "diagnostics_snapshot")
        add(
            "Sensing requests",
            storage="RAM",
            entries=diagnostics.get("request_entries"),
            bytes=diagnostics.get("request_bytes"),
            max_entries=diagnostics.get("request_entry_limit"),
            max_bytes=diagnostics.get("request_byte_limit"),
            counters={
                name: (diagnostics.get("requests") or {}).get("ram_" + name)
                for name in ("hits", "misses", "evictions")
            },
            diagnostics=diagnostics,
            note=(
                "RAM hits/misses include prefetch. A hit can still need first-time "
                "Angle or Detector preparation."
            ),
        )
        state_reader = getattr(service, "replay_cache_state", None)
        if callable(state_reader):
            state = state_reader()
            for title, entries, capacity in (
                ("Sensing metadata", "metadata_entries", None),
                (
                    "Selected sensing products",
                    "selected_product_entries",
                    "selected_product_capacity",
                ),
                ("Detector stage bundles", "detection_stage_entries", "detection_stage_capacity"),
                ("Validated sensing events", "validated_event_entries", "validated_event_capacity"),
            ):
                add(
                    title,
                    storage="RAM",
                    entries=getattr(state, entries),
                    max_entries=getattr(state, capacity) if capacity else None,
                )
        add(
            "Prepared sensing disk cache",
            storage="disk",
            max_bytes=MAX_DISK_BYTES,
            root=str(disk.root) if disk else None,
            counters={"hits": disk.hits if disk else None},
            note="On-demand disk budget; published bundles are separate; occupancy not scanned",
        )
        prepared = getattr(session, "preparation_summary", None)
        if disk is not None and getattr(disk, "_prepared_root", None) is not None:
            add(
                "Prepared recording bundle",
                storage="disk",
                root=str(disk._prepared_root),
                bytes=prepared.get("bundle_bytes") if prepared else None,
                note="Size recorded at preparation; no rescan or automatic total bundle limit",
            )
        history = getattr(session, "history", None)
        add(
            "Active recording overview",
            storage="RAM",
            entries=_length(getattr(history, "records", None)),
            max_entries=MAX_HISTORY_RESULTS,
            max_bytes=MAX_HISTORY_BYTES,
            note="Per-history bounds; byte limit covers numerical arrays; actual bytes not traversed",
        )
        sensing = {
            "session_identity": id(session),
            "source_identity": getattr(session, "source_identity", None),
            "requested_result": getattr(session, "pending_key", None),
            "result": _future_state(session.pending),
            "prefetch": _future_state(session._prefetch_pending),
            "history": _future_state(session.history_pending),
            "last_reader_progress": session.progress,
            "diagnostics": diagnostics,
            "figure_preparation_at_open": dict(prepared) if prepared else None,
        }
    panels = getattr(getattr(visualizer, "ui_manager", None), "panels", {})
    preparation = getattr(panels.get("sensing"), "preparation", None)
    sensing["preparation"] = _future_state(getattr(preparation, "future", None))
    sensing["preparation_progress"] = getattr(preparation, "progress", None)
    animation = getattr(visualizer, "animation_service", None)
    return {
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "scenario": str(getattr(visualizer, "current_scenario_path", "")),
        "scene_frame": getattr(visualizer, "animation_step", None),
        "memory": memory_snapshot(),
        "caches": rows,
        "sensing": sensing,
        "preload": {
            "frames": getattr(animation, "preload_frame_count", None),
            "complete": getattr(animation, "preloading_completed", None),
        },
        "notes": "Cache bytes can share ownership and are not process RSS. Asset inventories are last published. GPU VRAM is not sampled.",
    }


def _bytes(value: Any) -> str:
    if value is None:
        return "—"
    value = float(value)
    return f"{value / 2**30:.2f} GiB" if value >= 2**30 else f"{value / 2**20:.1f} MiB"


class CacheDebugWindow(QDialog):
    """Modeless diagnostic window whose explicit visibility owns all sampling."""

    def __init__(self, visualizer: Any) -> None:
        super().__init__(visualizer if isinstance(visualizer, QWidget) else None)
        self.visualizer = visualizer
        self._snapshot = None
        self._observed_session = None
        self.setWindowTitle("Cache and memory diagnostics")
        self.resize(1000, 720)
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.live = QCheckBox("Live sampling · 1 s")
        self.live.setChecked(True)
        self.live.toggled.connect(self._sync_sampling)
        controls.addWidget(self.live)
        refresh = QPushButton("Refresh once")
        refresh.clicked.connect(self.sample)
        controls.addWidget(refresh)
        self.copy_button = QPushButton("Copy report")
        self.copy_button.setEnabled(False)
        self.copy_button.clicked.connect(self.copy_report)
        controls.addWidget(self.copy_button)
        controls.addStretch()
        layout.addLayout(controls)
        memory_layout = QGridLayout()
        self.memory_labels = {}
        for row, (key, title) in enumerate(
            (
                ("process_rss_bytes", "Process RAM"),
                ("process_private_commit_bytes", "Process private commit"),
                ("system_used_bytes", "System RAM in use"),
                ("system_available_bytes", "System RAM available"),
                ("system_total_bytes", "System RAM total"),
                ("process_commit_headroom_bytes", "Process commit headroom (Windows)"),
                ("process_swap_bytes", "Process swap (Linux)"),
                ("system_swap_used_bytes", "System swap in use (Linux)"),
            )
        ):
            memory_layout.addWidget(QLabel(title), row // 2, (row % 2) * 2)
            value = self.memory_labels[key] = QLabel("—")
            memory_layout.addWidget(value, row // 2, (row % 2) * 2 + 1)
        layout.addLayout(memory_layout)
        self.cache_tree = QTreeWidget()
        self.cache_tree.setHeaderLabels(
            ["Cache", "Storage", "Entries / limit", "Bytes / limit", "Hits / misses / evictions"]
        )
        self.cache_tree.setRootIsDecorated(False)
        self.cache_tree.setAlternatingRowColors(True)
        layout.addWidget(self.cache_tree, 2)
        self.work = QPlainTextEdit()
        self.work.setReadOnly(True)
        self.work.setMaximumHeight(190)
        layout.addWidget(self.work)
        self.status = QLabel("Sampling starts only when this window is visible.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._sample_tick)

    def _visible(self) -> bool:
        return self.isVisible() and not self.isMinimized()

    def _observe_session(self, enabled: bool) -> None:
        session = dashboard_session(self.visualizer) if enabled else None
        if session is self._observed_session:
            return
        if self._observed_session is not None:
            setter = getattr(self._observed_session, "set_diagnostics_enabled", None)
            if setter is not None:
                setter(False)
        self._observed_session = session
        setter = getattr(session, "set_diagnostics_enabled", None)
        if setter is not None:
            setter(True)

    def _sync_sampling(self, *_args: Any) -> None:
        enabled = self._visible() and self.live.isChecked()
        self._observe_session(enabled)
        if enabled:
            self.sample()
            self.timer.start()
        else:
            self.timer.stop()
            self._update_status()

    def _sample_tick(self) -> None:
        """Ignore a queued timeout after live sampling has been disabled."""
        if self.live.isChecked():
            self.sample()

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self._sync_sampling()

    def hideEvent(self, event: Any) -> None:
        self.timer.stop()
        self._observe_session(False)
        super().hideEvent(event)

    def changeEvent(self, event: Any) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            self._sync_sampling()

    def shutdown(self) -> None:
        """Stop diagnostics before the owning panel is disposed."""
        self.timer.stop()
        self._observe_session(False)
        self.close()

    def sample(self) -> None:
        """Take one explicit visible snapshot; hidden timeout deliveries do no work."""
        if not self._visible():
            return
        self._observe_session(self.live.isChecked())
        self._snapshot = snapshot = cache_debug_snapshot(self.visualizer)
        for key, label in self.memory_labels.items():
            label.setText(_bytes(snapshot["memory"].get(key)))
        self.cache_tree.clear()
        for row in snapshot["caches"]:
            entries = "—" if row.get("entries") is None else str(row["entries"])
            if row.get("max_entries") is not None:
                entries += f" / {row['max_entries']}"
            size = _bytes(row.get("bytes"))
            if row.get("max_bytes") is not None:
                size += " / " + _bytes(row["max_bytes"])
            counters = row.get("counters", row)
            counts = " / ".join(
                str(value) if value is not None else "—"
                for value in (
                    counters.get("hits", counters.get("hit")),
                    counters.get("misses", counters.get("miss")),
                    counters.get("evictions", counters.get("eviction")),
                )
            )
            item = QTreeWidgetItem([row["name"], row["storage"], entries, size, counts])
            item.setToolTip(0, json.dumps(row, indent=2))
            self.cache_tree.addTopLevelItem(item)
        for column in range(4):
            self.cache_tree.resizeColumnToContents(column)
        sensing = snapshot["sensing"]
        diagnostics = sensing.get("diagnostics", {})
        reader = diagnostics.get("reader") or {}
        requests = diagnostics.get("requests") or {}
        lines = [
            f"Scene frame: {snapshot.get('scene_frame')} · dashboard session: {sensing.get('session_identity', 'not open')}",
            f"Result: {sensing.get('result', 'idle')} · prefetch: {sensing.get('prefetch', 'idle')} · overview: {sensing.get('history', 'idle')}",
            f"Initial preparation: {sensing.get('preparation', 'idle')} · {sensing.get('preparation_progress') or ''}",
            f"Requested result: {sensing.get('requested_result') or 'none'}",
            f"Last reader report: {sensing.get('last_reader_progress') or 'none'}",
            f"Observed index builds: {reader.get('index_builds', '—')} · disk reuses: {reader.get('index_disk_hits', '—')} · dense reads: {reader.get('selected_dense_loads', '—')}",
        ]
        prepared = sensing.get("figure_preparation_at_open")
        if prepared:
            lines.append(
                f"Figure requests resident at opening: {prepared['retained_requests']}/"
                f"{prepared['requested_requests']} · current occupancy is shown above"
            )
        for name, title in (
            ("last_queue_ms", "Last worker queue"),
            ("last_result_ms", "Last result preparation"),
            ("last_request_ms", "Last complete request"),
        ):
            value = requests.get(name)
            lines.append(f"{title}: {value:.1f} ms" if value is not None else f"{title}: —")
        self.work.setPlainText("\n".join(lines))
        self._update_status()
        self.copy_button.setEnabled(True)

    def _update_status(self) -> None:
        if self._snapshot is None:
            return
        snapshot = self._snapshot
        mode = "Live" if self._visible() and self.live.isChecked() else "Paused"
        self.status.setText(
            f"{mode} · last sample {snapshot.get('sampled_at', 'unavailable')}. "
            + snapshot["notes"]
            + " Sensing request/reader counters measure only time while live diagnostics are enabled."
            + (
                " " + snapshot["memory"]["unavailable"]
                if "unavailable" in snapshot["memory"]
                else ""
            )
        )

    def copy_report(self) -> None:
        """Copy the latest sample without triggering another read or refresh."""
        if self._snapshot is not None:
            QApplication.clipboard().setText(json.dumps(self._snapshot, indent=2))
