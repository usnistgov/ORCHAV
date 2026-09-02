"""Stable Qt workspace and embedded renderer host for the visualizer."""

from __future__ import annotations

from enum import Enum
from typing import Any

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class ViewportState(str, Enum):
    """User-visible lifecycle state of the normal visualization viewport."""

    EMPTY = "empty"
    LOADING = "loading"
    ACTIVE = "active"
    ERROR = "error"


class EmbeddedViewportHost(QWidget):
    """Persistent Qt parent for one renderer canvas at a time.

    The host remains attached to the main-window widget tree while renderer
    sessions are created and destroyed. Native canvases are therefore never
    reparented after construction.
    """

    open_requested = Signal()
    retry_requested = Signal()
    logical_size_changed = Signal(int, int)
    screen_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("embeddedViewportHost")
        self.setMinimumSize(320, 240)
        self._state = ViewportState.EMPTY
        self._bound_window_handle: Any = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self._stack = QStackedWidget(self)
        root.addWidget(self._stack)

        self._empty_page = self._message_page(
            "Open a scenario to begin visualization.",
            primary_label="Open Scenario…",
            primary_signal=self.open_requested,
        )

        self._loading_page = QWidget(self._stack)
        loading_layout = QVBoxLayout(self._loading_page)
        loading_layout.addStretch()
        self._loading_label = QLabel("Loading scenario…", self._loading_page)
        self._loading_label.setAlignment(Qt.AlignCenter)
        self._loading_label.setWordWrap(True)
        loading_layout.addWidget(self._loading_label)
        self._loading_progress = QProgressBar(self._loading_page)
        self._loading_progress.setRange(0, 0)
        loading_layout.addWidget(self._loading_progress)
        loading_layout.addStretch()

        self._canvas_page = QWidget(self._stack)
        self._canvas_page.setObjectName("embeddedViewportCanvasParent")
        canvas_layout = QVBoxLayout(self._canvas_page)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self._error_page = QWidget(self._stack)
        error_layout = QVBoxLayout(self._error_page)
        error_layout.addStretch()
        self._error_label = QLabel("The scenario could not be displayed.", self._error_page)
        self._error_label.setAlignment(Qt.AlignCenter)
        self._error_label.setWordWrap(True)
        error_layout.addWidget(self._error_label)
        error_buttons = QHBoxLayout()
        error_buttons.addStretch()
        retry_button = QPushButton("Retry", self._error_page)
        retry_button.clicked.connect(self.retry_requested.emit)
        error_buttons.addWidget(retry_button)
        open_button = QPushButton("Open Another…", self._error_page)
        open_button.clicked.connect(self.open_requested.emit)
        error_buttons.addWidget(open_button)
        error_buttons.addStretch()
        error_layout.addLayout(error_buttons)
        error_layout.addStretch()

        for page in (
            self._empty_page,
            self._loading_page,
            self._canvas_page,
            self._error_page,
        ):
            self._stack.addWidget(page)
        self.set_state(ViewportState.EMPTY)

    @property
    def state(self) -> ViewportState:
        """Return the currently displayed viewport state."""

        return self._state

    @property
    def canvas_parent(self) -> QWidget:
        """Return the final Qt parent supplied to an embedded renderer."""

        return self._canvas_page

    def set_state(self, state: ViewportState | str, message: str | None = None) -> None:
        """Display one lifecycle page and optionally update its message."""

        state = ViewportState(state)
        self._state = state
        if state is ViewportState.LOADING:
            if message:
                self._loading_label.setText(message)
            self._stack.setCurrentWidget(self._loading_page)
        elif state is ViewportState.ACTIVE:
            self._stack.setCurrentWidget(self._canvas_page)
        elif state is ViewportState.ERROR:
            if message:
                self._error_label.setText(message)
            self._stack.setCurrentWidget(self._error_page)
        else:
            self._stack.setCurrentWidget(self._empty_page)

    def set_loading_progress(self, step: int, total: int, message: str) -> None:
        """Update the integrated loading page without opening a dialog."""

        self._loading_label.setText(message)
        self._loading_progress.setRange(0, max(0, int(total)))
        self._loading_progress.setValue(max(0, min(int(step), int(total))))
        self.set_state(ViewportState.LOADING)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        """Publish positive canvas dimensions to the active renderer."""

        super().resizeEvent(event)
        size = self._canvas_page.size()
        if size.width() > 0 and size.height() > 0:
            self.logical_size_changed.emit(size.width(), size.height())

    def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        """Bind the persistent host to its current top-level window once."""

        super().showEvent(event)
        handle = self.window().windowHandle()
        if handle is self._bound_window_handle or handle is None:
            return
        if self._bound_window_handle is not None:
            try:
                self._bound_window_handle.screenChanged.disconnect(self.screen_changed.emit)
            except (RuntimeError, TypeError):
                pass
        self._bound_window_handle = handle
        handle.screenChanged.connect(self.screen_changed.emit)

    def _message_page(
        self,
        message: str,
        *,
        primary_label: str,
        primary_signal: Any,
    ) -> QWidget:
        page = QWidget(self._stack)
        layout = QVBoxLayout(page)
        layout.addStretch()
        label = QLabel(message, page)
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        layout.addWidget(label)
        row = QHBoxLayout()
        row.addStretch()
        button = QPushButton(primary_label, page)
        button.clicked.connect(primary_signal.emit)
        row.addWidget(button)
        row.addStretch()
        layout.addLayout(row)
        layout.addStretch()
        return page


class VisualizationWorkspace(QWidget):
    """Normal controls-and-viewport page retained across mode switches."""

    _SPLITTER_SETTINGS_KEY = "visualizer/workspace_splitter_state"

    def __init__(
        self,
        controls: QWidget,
        *,
        embedded: bool,
        parent: QWidget | None = None,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("visualizationWorkspace")
        self._settings = settings or QSettings()
        self._embedded = bool(embedded)
        self.viewport_host = EmbeddedViewportHost(self)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.splitter = QSplitter(Qt.Horizontal, self)
        self.splitter.setObjectName("visualizationWorkspaceSplitter")
        self.splitter.addWidget(controls)
        if embedded:
            self.splitter.addWidget(self.viewport_host)
            self.splitter.setStretchFactor(0, 0)
            self.splitter.setStretchFactor(1, 1)
            if not self._restore_splitter_state():
                self.splitter.setSizes((420, 1100))
        else:
            self.viewport_host.hide()
        root.addWidget(self.splitter)

    def save_settings(self) -> None:
        """Persist only visual shell geometry, never scenario state."""

        if self._embedded:
            self._settings.setValue(self._SPLITTER_SETTINGS_KEY, self.splitter.saveState())

    def _restore_splitter_state(self) -> bool:
        state = self._settings.value(self._SPLITTER_SETTINGS_KEY)
        return state is not None and bool(self.splitter.restoreState(state))
