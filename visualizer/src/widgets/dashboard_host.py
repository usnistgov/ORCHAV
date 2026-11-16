"""Move one dashboard between its panel and a persistent floating shell."""

from PySide6.QtCore import QEvent, QSettings, Signal
from PySide6.QtWidgets import QLabel, QMainWindow, QPushButton, QScrollArea, QVBoxLayout, QWidget
from shiboken6 import isValid


class _DashboardWindow(QMainWindow):
    return_requested = Signal()

    def closeEvent(self, event):
        # Closing the shell returns its content; it must not retire the plots.
        self.return_requested.emit()
        event.accept()


class DashboardHost(QWidget):
    """Own placement only; each dashboard's caller controls data and refresh timing."""

    activated = Signal()
    returned = Signal()

    def __init__(self, parent=None, *, title, settings_key, settings=None):
        super().__init__(parent)
        self._title = title
        self._settings_key = settings_key
        self._settings = settings if settings is not None else QSettings()
        self._observed_window = None
        self.dashboard = None
        self.floating_window = None
        self._disposed = False
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.placeholder = QWidget(self)
        placeholder_layout = QVBoxLayout(self.placeholder)
        self.notice = QLabel("Loading dashboard…")
        self.notice.setWordWrap(True)
        placeholder_layout.addWidget(self.notice)
        self.return_button = QPushButton("Return dashboard to panel")
        self.return_button.clicked.connect(self.dock)
        self.return_button.hide()
        placeholder_layout.addWidget(self.return_button)
        self._layout.addWidget(self.placeholder)

    def set_dashboard(self, dashboard):
        self.dashboard = dashboard
        dashboard.destroyed.connect(lambda: self._dashboard_destroyed(dashboard))
        dashboard._hosted = True
        dashboard.placement_button.show()
        dashboard.placement_button.clicked.connect(self.toggle_placement)
        self._layout.addWidget(dashboard)
        self.placeholder.hide()
        dashboard.show()

    def _dashboard_destroyed(self, dashboard):
        if self.dashboard is dashboard:
            self.dashboard = None

    def toggle_placement(self):
        if self.is_detached():
            self.dock()
        else:
            self.detach()

    def is_detached(self):
        return (
            self.floating_window is not None
            and self.dashboard is not None
            and self.floating_window.centralWidget().widget() is self.dashboard
        )

    def detach(self):
        if self.dashboard is None or self._disposed:
            return
        if self.is_detached():
            self.floating_window.showNormal()
            self.floating_window.raise_()
            self.floating_window.activateWindow()
            return
        if self.floating_window is None:
            self.floating_window = _DashboardWindow(self.window())
            self.floating_window.setWindowTitle(self._title)
            self.floating_window.resize(1100, 800)
            geometry = self._settings.value(self._settings_key)
            if geometry is not None:
                self.floating_window.restoreGeometry(geometry)
            self.floating_window.installEventFilter(self)
            self.floating_window.return_requested.connect(self.dock)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            self.floating_window.setCentralWidget(scroll)
        scroll = self.floating_window.centralWidget()
        self._layout.removeWidget(self.dashboard)
        scroll.setWidget(self.dashboard)
        self.dashboard.placement_button.setText("Return to panel")
        self.notice.setText("The dashboard is open in a separate window.")
        self.return_button.show()
        self.placeholder.show()
        self.dashboard.show()
        self.floating_window.show()
        self.floating_window.raise_()
        self.floating_window.activateWindow()
        self.activated.emit()

    def dock(self, *, reveal=True):
        if self.dashboard is None or self._disposed:
            return
        detached = self.is_detached()
        if self.floating_window is not None:
            if detached:
                self._save_geometry()
            self.floating_window.centralWidget().takeWidget()
            self.floating_window.hide()
        self._layout.addWidget(self.dashboard)
        self.dashboard.placement_button.setText("Pop out")
        self.placeholder.hide()
        self.dashboard.show()
        if detached and reveal:
            self.returned.emit()
        elif self.isVisible():
            self.activated.emit()

    def suspend(self):
        """Return floating analysis before a volume replaces recording controls."""
        self.dock(reveal=False)

    def dispose(self):
        """Release the floating shell; dashboard cleanup belongs to the panel."""
        self._disposed = True
        if self.floating_window is not None:
            if self.is_detached():
                self._save_geometry()
            self.floating_window.centralWidget().takeWidget()
            self.floating_window.hide()
            self.floating_window.deleteLater()
            self.floating_window = None
        if self.dashboard is not None:
            self.dashboard.setParent(self)
        self.dashboard = None

    def showEvent(self, event):
        super().showEvent(event)
        window = self.window()
        if self._observed_window is not window:
            if self._observed_window is not None:
                self._observed_window.removeEventFilter(self)
            self._observed_window = window
            window.installEventFilter(self)
        if not self._disposed:
            self.activated.emit()

    def _save_geometry(self):
        self._settings.setValue(self._settings_key, self.floating_window.saveGeometry())

    def eventFilter(self, watched, event):
        # PySide can expose an unrelated or expired wrapper at this boundary.
        # Do not inspect it or forward it to QObject's typed, no-op filter.
        if not isinstance(event, QEvent) or not isValid(event):
            return False
        if event.type() == QEvent.WindowStateChange and not self._disposed:
            self.activated.emit()
        return False


def dashboard_is_visible(widget):
    """A child widget stays Qt-visible while its containing window is minimized."""
    if not widget.isVisible():
        return False
    window_getter = getattr(widget, "window", None)
    return not (callable(window_getter) and window_getter().isMinimized())
