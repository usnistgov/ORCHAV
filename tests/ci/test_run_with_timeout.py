"""Tests for the release-smoke process timeout wrapper."""

from __future__ import annotations

import sys

from scripts.ci.run_with_timeout import TIMEOUT_EXIT_CODE, run_with_timeout


def test_timeout_wrapper_returns_child_status() -> None:
    assert run_with_timeout([sys.executable, "-c", "raise SystemExit(7)"], 10) == 7


def test_timeout_wrapper_bounds_a_long_running_child() -> None:
    command = [sys.executable, "-c", "import time; time.sleep(30)"]

    assert run_with_timeout(command, 0.1) == TIMEOUT_EXIT_CODE
