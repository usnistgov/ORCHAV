"""Shared Qt process ownership for explicit map and OSM scene requests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QObject, QProcess, QTimer, Signal


def osm_cache_root() -> Path:
    return Path(os.environ.get("ORCHAV_CACHE_DIR", str(Path.home() / ".orchav" / "cache")))


class OsmJob(QObject):
    progress = Signal(str)
    finished = Signal(object)

    def __init__(self, operation, config, parent=None, *, cache_root=None):
        super().__init__(parent)
        self._python = (
            os.environ.get("ORCHAV_OSM_PYTHON", sys.executable)
            if operation == "build"
            else sys.executable
        )
        self.root = Path(cache_root or osm_cache_root()) / "authoring-osm" / uuid4().hex
        self.root.mkdir(parents=True, exist_ok=False)
        self.request = self.root / "request.json"
        self.request.write_text(
            json.dumps({"operation": operation, "config": config}), encoding="utf-8"
        )
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(Path(__file__).resolve().parents[3]))
        self.process.setStandardErrorFile(str(self.root / "worker.log"))
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        self._pending = bytearray()
        self._result = None
        self._done = False
        self.cancelled = False
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._kill)
        self._heartbeat = QTimer(self)
        self._heartbeat.setInterval(1000)
        self._heartbeat.timeout.connect(self._tick)
        self._message = "Starting OSM worker…"
        self._started = time.monotonic()

    @property
    def running(self):
        return not self._done

    def start(self):
        self._started = time.monotonic()
        self._heartbeat.start()
        source_root = str(Path(__file__).resolve().parents[3])
        bootstrap = (
            "import sys,runpy;sys.path.insert(0,sys.argv.pop(1));"
            "runpy.run_module('visualizer.src.authoring.osm_worker',run_name='__main__')"
        )
        self.process.start(self._python, ["-u", "-c", bootstrap, source_root, str(self.request)])

    def _tick(self):
        self.progress.emit(f"{self._message}\nElapsed: {time.monotonic() - self._started:.0f} s")

    def _read(self):
        self._pending.extend(bytes(self.process.readAllStandardOutput()))
        while b"\n" in self._pending:
            line, _, self._pending = self._pending.partition(b"\n")
            try:
                value = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(value, dict):
                continue
            if value.get("event") == "progress":
                detail = value.get("detail", {})
                message = detail.get("message") or " · ".join(
                    str(detail[k]) for k in ("stage", "layer", "outcome") if detail.get(k)
                )
                if detail.get("cache"):
                    message += " · " + (
                        "reused cached data" if detail["cache"] == "hit" else "downloaded"
                    )
                self._message = message or "Preparing OSM scene…"
                self._tick()
            elif value.get("event") in {"complete", "failed"}:
                self._result = value

    def _error(self, error):
        if error == QProcess.FailedToStart:
            self._finish({"event": "failed", "error": self.process.errorString()})

    def _finished(self, code, _status):
        self._read()
        result = self._result or {
            "event": "failed",
            "error": f"OSM worker exited ({code}). See {self.root / 'worker.log'}",
        }
        if code and result.get("event") == "complete":
            result = {"event": "failed", "error": f"OSM worker exited unexpectedly ({code})."}
        self._finish(result)

    def _finish(self, result):
        if self._done:
            return
        self._done = True
        self._kill_timer.stop()
        self._heartbeat.stop()
        if self.cancelled:
            result = {"event": "cancelled", "error": "Cancelled. The existing scene is unchanged."}
        self.finished.emit(result)

    def cancel(self):
        if self._done:
            return
        self.cancelled = True
        self._message = "Cancelling…"
        (self.root / "cancel").touch()
        self._kill_timer.start(2500)

    def _kill(self):
        if self.process.state() == QProcess.NotRunning:
            return
        # The generator's HTTP helper is a child process; terminate this owned tree too.
        pid = int(self.process.processId())
        if os.name == "nt" and pid:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    timeout=2,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
        else:
            self.process.kill()

    def close(self):
        self.cancel()
        if self.process.state() != QProcess.NotRunning and not self.process.waitForFinished(2700):
            self._kill()
            self.process.waitForFinished(1500)
