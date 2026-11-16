"""Tests for the release-smoke process timeout wrapper."""

from __future__ import annotations

import os
import signal
import sys
import time

import psutil
import pytest

from scripts.ci import run_with_timeout as timeout_module
from scripts.ci.run_with_timeout import TIMEOUT_EXIT_CODE, run_with_timeout


def test_timeout_wrapper_returns_child_status() -> None:
    assert run_with_timeout([sys.executable, "-c", "raise SystemExit(7)"], 10) == 7


def test_timeout_wrapper_bounds_a_long_running_child() -> None:
    command = [sys.executable, "-c", "import time; time.sleep(30)"]

    assert run_with_timeout(command, 0.1) == TIMEOUT_EXIT_CODE


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group signals")
def test_timeout_kills_resistant_descendant_after_leader_exits(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(timeout_module, "TERMINATION_GRACE_S", 0.2)
    pid_path = tmp_path / "descendant.pid"
    descendant = (
        "import os, pathlib, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )
    leader = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {descendant!r}]); "
        "time.sleep(30)"
    )
    child = None
    try:
        assert run_with_timeout([sys.executable, "-c", leader], 2) == TIMEOUT_EXIT_CODE
        child = psutil.Process(int(pid_path.read_text())) if pid_path.exists() else None
        assert pid_path.exists(), "descendant did not start before the timeout"
        deadline = time.monotonic() + 2
        while child is not None and time.monotonic() < deadline:
            if not child.is_running() or child.status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(0.01)
        assert child is None or not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass
    finally:
        if pid_path.exists():
            try:
                os.kill(int(pid_path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
