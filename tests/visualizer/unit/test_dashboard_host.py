"""The placement host observes window state without consuming native events."""

import pytest
from PySide6.QtCore import QEvent, QSettings, Qt
from PySide6.QtGui import QStandardItem, QWindowStateChangeEvent
from PySide6.QtWidgets import QApplication, QWidget
from shiboken6 import delete

from visualizer.src.widgets.dashboard_host import DashboardHost


def _host(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "host.ini"), QSettings.IniFormat)
    host = DashboardHost(title="Result graphs", settings_key="probe", settings=settings)
    qtbot.addWidget(host)
    return host


@pytest.mark.parametrize("kind", ["unrelated_item", "expired_event"])
def test_host_filter_ignores_unusable_event_wrappers(qtbot, tmp_path, kind):
    host = _host(qtbot, tmp_path)
    window = QWidget()
    qtbot.addWidget(window)
    activations = []
    host.activated.connect(lambda: activations.append(True))
    if kind == "unrelated_item":
        event = QStandardItem("Unexpected event wrapper")
    else:
        event = QEvent(QEvent.WindowStateChange)
        delete(event)
    assert host.eventFilter(window, event) is False
    assert not activations


@pytest.mark.parametrize("disposed", [False, True])
def test_host_observes_state_changes_without_consuming_events(qtbot, tmp_path, disposed):
    class ObservedWindow(QWidget):
        def __init__(self):
            self.delivered = []
            super().__init__()

        def event(self, event):
            if event.type() in {QEvent.WindowStateChange, QEvent.User}:
                self.delivered.append(event.type())
            return super().event(event)

    host = _host(qtbot, tmp_path)
    window = ObservedWindow()
    qtbot.addWidget(window)
    activations = []
    host.activated.connect(lambda: activations.append(True))
    window.installEventFilter(host)
    if disposed:
        host.dispose()
    QApplication.sendEvent(window, QWindowStateChangeEvent(Qt.WindowNoState))
    QApplication.sendEvent(window, QEvent(QEvent.User))
    assert window.delivered == [QEvent.WindowStateChange, QEvent.User]
    assert len(activations) == (0 if disposed else 1)
