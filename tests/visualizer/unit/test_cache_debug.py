"""Opt-in diagnostics never load recordings or sample while hidden/disabled."""

import ctypes
import json
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

from visualizer.src.diagnostics import cache_debug as module
from visualizer.src.panels.performance_panel import PerformancePanel


def _sample():
    return {
        "memory": {"process_rss_bytes": 1234, "system_available_bytes": 2**30},
        "caches": [],
        "sensing": {},
        "notes": "No source reads",
    }


def test_panel_construction_does_not_create_or_sample_debug_window(qtbot, monkeypatch):
    def unexpected(*_args):
        raise AssertionError("Disabled diagnostics sampled")

    monkeypatch.setattr(module, "cache_debug_snapshot", unexpected)
    panel = PerformancePanel(SimpleNamespace())
    group = panel.create_panel()
    qtbot.addWidget(group)
    assert panel._cache_debug_window is None
    assert panel._refresh_timer is None
    panel.cleanup()


def test_sampling_and_instrumentation_follow_explicit_visibility(qtbot, monkeypatch):
    observations, samples = [], []
    session = SimpleNamespace(set_diagnostics_enabled=observations.append)
    owner = SimpleNamespace(
        ui_manager=SimpleNamespace(panels={"sensing": SimpleNamespace(_dashboard_session=session)})
    )
    monkeypatch.setattr(module, "cache_debug_snapshot", lambda _: samples.append(True) or _sample())
    window = module.CacheDebugWindow(owner)
    qtbot.addWidget(window)
    assert not window.timer.isActive()
    assert not samples and not observations
    window.show()
    assert window.timer.isActive()
    assert observations == [True]
    assert len(samples) == 1

    window.live.setChecked(False)
    assert not window.timer.isActive()
    assert observations == [True, False]
    window.timer.timeout.emit()  # A queued timeout after disabling must do nothing.
    assert len(samples) == 1
    window.sample()  # Explicit refresh remains available without live instrumentation.
    assert len(samples) == 2
    assert observations == [True, False]

    window.hide()
    window.sample()
    window.timer.timeout.emit()
    assert len(samples) == 2
    window.show()
    assert not window.timer.isActive()
    assert len(samples) == 2
    window.live.setChecked(True)
    assert window.timer.isActive()
    assert len(samples) == 3
    window.showMinimized()
    qtbot.waitUntil(lambda: not window.timer.isActive())
    window.timer.timeout.emit()
    assert len(samples) == 3
    window.showNormal()
    qtbot.waitUntil(window.timer.isActive)
    window.close()
    assert not window.timer.isActive()
    assert observations[-1] is False


def test_session_replacement_stops_previous_instrumentation_and_copy_is_passive(qtbot, monkeypatch):
    first, second, samples = [], [], []
    panel = SimpleNamespace(
        _dashboard_session=SimpleNamespace(set_diagnostics_enabled=first.append)
    )
    owner = SimpleNamespace(ui_manager=SimpleNamespace(panels={"sensing": panel}))
    monkeypatch.setattr(module, "cache_debug_snapshot", lambda _: samples.append(True) or _sample())
    window = module.CacheDebugWindow(owner)
    qtbot.addWidget(window)
    window.show()
    panel._dashboard_session = SimpleNamespace(set_diagnostics_enabled=second.append)
    window.sample()
    assert first == [True, False]
    assert second == [True]
    window.copy_report()
    assert len(samples) == 2
    assert json.loads(module.QApplication.clipboard().text()) == _sample()
    window.shutdown()
    assert second == [True, False]
    assert not window.timer.isActive()


def test_performance_cleanup_stops_open_debug_window(qtbot, monkeypatch):
    samples = []
    monkeypatch.setattr(module, "cache_debug_snapshot", lambda _: samples.append(True) or _sample())
    panel = PerformancePanel(SimpleNamespace())
    group = panel.create_panel()
    qtbot.addWidget(group)
    panel.widgets["cache_debug_button"].click()
    window = panel._cache_debug_window
    assert window.timer.isActive()
    panel.cleanup()
    assert not window.timer.isActive()
    assert panel._cache_debug_window is None
    assert len(samples) == 1


def test_snapshot_reads_only_accounting_and_does_not_touch_payloads_or_disk(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("Diagnostics attempted a data read or disk scan")

    class Payloads(dict):
        values = items = __iter__ = unexpected

    monkeypatch.setattr(Path, "stat", unexpected)
    monkeypatch.setattr(Path, "glob", unexpected)
    monkeypatch.setattr(module, "memory_snapshot", lambda: {"process_rss_bytes": 100})
    monkeypatch.setattr(
        module,
        "cache_telemetry_snapshot",
        lambda: {
            "texture_decode": {
                "current_entries": 2,
                "current_bytes": 12,
                "counters": {"hit": 3},
                "elapsed_ms": {},
                "root": None,
            }
        },
    )
    pending = Future()
    pending.set_running_or_notify_cancel()
    diagnostic_calls = []
    session = SimpleNamespace(
        _request_cache=Payloads({1: object()}),
        _request_cache_bytes=23,
        service=SimpleNamespace(
            recording_cache=SimpleNamespace(root=Path("cache"), hits=4, unchanged=unexpected)
        ),
        pending=pending,
        _prefetch_pending=None,
        history_pending=None,
        progress=("Reading", 2, 5),
        pending_key=(64, "angular"),
        source_identity=(1, "source"),
        diagnostics_snapshot=lambda: diagnostic_calls.append(True)
        or {
            "request_entries": 1,
            "request_bytes": 23,
            "request_entry_limit": 512,
            "request_byte_limit": 512 * 2**20,
            "requests": {"ram_hits": 8, "ram_misses": 2, "ram_evictions": 1},
        },
        request=unexpected,
    )
    owner = SimpleNamespace(
        ui_manager=SimpleNamespace(panels={"sensing": SimpleNamespace(_dashboard_session=session)}),
        cache_service=SimpleNamespace(
            get_cache_telemetry=unexpected,
            frame_cache=SimpleNamespace(
                size=2, max_size=5, stats=SimpleNamespace(hits=7, misses=2, evictions=1)
            ),
        ),
        mpc_core=SimpleNamespace(
            _canon_cache=Payloads({1: object()}), _canon_bytes_used=10, _canon_max_bytes=100
        ),
        mpc_view_cache=Payloads({1: object()}),
        frame_source=SimpleNamespace(load_frame=unexpected),
    )
    snapshot = module.cache_debug_snapshot(owner)
    rows = {row["name"]: row for row in snapshot["caches"]}
    assert rows["Sensing requests"]["bytes"] == 23
    assert rows["Sensing requests"]["counters"] == {"hits": 8, "misses": 2, "evictions": 1}
    assert rows["Canonical MPC data"]["bytes"] == 10
    assert rows["Scene frames"]["counters"]["hits"] == 7
    assert snapshot["sensing"]["result"] == "running"
    assert len(diagnostic_calls) == 1
    json.dumps(snapshot)
    session.service.recording_cache._prepared_root = Path("cache/recordings/bundle_example")
    session.preparation_summary = {
        "requested_requests": 80,
        "retained_requests": 80,
        "bundle_bytes": 12345,
    }
    snapshot = module.cache_debug_snapshot(owner)
    bundle = next(row for row in snapshot["caches"] if row["name"] == "Prepared recording bundle")
    assert bundle["bytes"] == 12345
    assert snapshot["sensing"]["figure_preparation_at_open"] == session.preparation_summary
    json.dumps(snapshot)


def test_linux_memory_units_and_available_ram(monkeypatch):
    content = {
        "/proc/self/status": "VmRSS:\t2048 kB\nVmSwap:\t4 kB\nThreads:\t5\n",
        "/proc/meminfo": "MemTotal: 8192 kB\nMemAvailable: 4096 kB\nSwapTotal: 1024 kB\nSwapFree: 768 kB\n",
    }
    monkeypatch.setattr(Path, "read_text", lambda path: content[str(path)])
    monkeypatch.setattr(module.sys, "platform", "linux")
    result = module.memory_snapshot()
    assert result["process_rss_bytes"] == 2**21
    assert result["system_used_bytes"] == 2**22
    assert result["system_swap_used_bytes"] == 256 * 1024
    assert "process_private_commit_bytes" not in result


def test_windows_memory_structures_and_pointer_width(monkeypatch):
    class Function:
        def __init__(self, callback):
            self.callback = callback

        def __call__(self, *args):
            return self.callback(*args)

    def system(pointer):
        data = pointer._obj
        assert data.length == ctypes.sizeof(data) == 64
        assert type(data).total.offset == 8
        data.total, data.available, data.commit_available = 8 * 2**30, 2 * 2**30, 512 * 2**20
        return 1

    def process(handle, pointer, size):
        assert handle == 42
        data = pointer._obj
        assert size == data.size == ctypes.sizeof(data)
        assert type(data).rss.offset == 8 + ctypes.sizeof(ctypes.c_size_t)
        data.rss, data.private = 3 * 2**30, 7 * 2**30
        return 1

    kernel = SimpleNamespace(
        GetCurrentProcess=Function(lambda: 42), GlobalMemoryStatusEx=Function(system)
    )
    psapi = SimpleNamespace(GetProcessMemoryInfo=Function(process))
    monkeypatch.setattr(
        ctypes, "WinDLL", lambda name, **_: kernel if name == "kernel32" else psapi, raising=False
    )
    result = module._windows_memory()
    assert result["process_rss_bytes"] == 3 * 2**30
    assert result["process_private_commit_bytes"] == 7 * 2**30
    assert result["system_available_bytes"] == 2 * 2**30
    assert result["process_commit_headroom_bytes"] == 512 * 2**20
    assert kernel.GetCurrentProcess.restype is ctypes.c_void_p


@pytest.mark.parametrize("platform_name", ["win32", "linux", "unsupported"])
def test_memory_unavailable_is_explicit_and_does_not_report_zero(monkeypatch, platform_name):
    def unavailable():
        raise OSError("counter unavailable")

    monkeypatch.setattr(module.sys, "platform", platform_name)
    monkeypatch.setattr(module, "_linux_memory", unavailable)
    monkeypatch.setattr(module, "_windows_memory", unavailable)
    assert set(module.memory_snapshot()) == {"unavailable"}
