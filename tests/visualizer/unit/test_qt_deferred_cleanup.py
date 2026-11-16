"""Test-session Qt cleanup retires closed windows and preserves active owners."""

import os
from pathlib import Path

pytest_plugins = ["pytester"]


def test_closed_window_is_deleted_between_tests_without_deleting_active_widget(
    pytester, monkeypatch
):
    root = Path(__file__).resolve().parents[3]
    # Widget ownership and deferred deletion do not require a window-system connection.
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")]))
    pytester.makeconftest('pytest_plugins = ["tests.visualizer.conftest"]')
    pytester.makepyfile("""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget
from shiboken6 import isValid

app = QApplication.instance() or QApplication([])
persistent = QWidget()
closed = None
child = None


def test_close_requests_deletion():
    global closed, child
    closed = QWidget()
    closed.setAttribute(Qt.WA_DeleteOnClose)
    child = QWidget(closed)
    closed.close()
    assert isValid(closed)
    assert isValid(persistent)


def test_next_test_has_no_closed_window_hierarchy():
    assert not isValid(closed)
    assert not isValid(child)
    assert isValid(persistent)
    assert QApplication.instance() is app
""")
    result = pytester.runpytest_subprocess("--no-cov", "-q")
    result.assert_outcomes(passed=2)
