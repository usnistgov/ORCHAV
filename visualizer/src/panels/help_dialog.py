"""Keyboard shortcut cheat-sheet dialog."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..app.shortcuts import shortcut_rows

SHORTCUTS: list[tuple[str, str]] = shortcut_rows()


class HelpDialog(QDialog):
    """Modal dialog displaying a two-column table of keyboard shortcuts."""

    def __init__(self, parent=None) -> None:
        """Build the shortcut table from the shared shortcut registry."""
        super().__init__(parent)
        self.setWindowTitle("Help and Controls")
        self.resize(640, 450)
        flags = self.windowFlags()
        flags |= Qt.WindowSystemMenuHint | Qt.WindowCloseButtonHint
        flags &= ~Qt.WindowContextHelpButtonHint
        self.setWindowFlags(flags)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        table = QTableWidget(len(SHORTCUTS), 2)
        table.setObjectName("applicationShortcuts")
        table.setHorizontalHeaderLabels(["Shortcut", "Action"])
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setFocusPolicy(Qt.NoFocus)
        table.setAlternatingRowColors(True)

        for row, (key, description) in enumerate(SHORTCUTS):
            key_item = QTableWidgetItem(key)
            key_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row, 0, key_item)
            table.setItem(row, 1, QTableWidgetItem(description))

        tabs = QTabWidget()
        general = QWidget()
        general_layout = QVBoxLayout(general)
        general_layout.addWidget(table)
        note = QLabel(
            "Focused controls keep their own keyboard actions. "
            "The mouse wheel adjusts the control under the pointer."
        )
        note.setWordWrap(True)
        general_layout.addWidget(note)
        tabs.addTab(general, "Application")
        for title, rows in (
            (
                "Pygfx viewport",
                [
                    ("Left drag", "Orbit camera; look around in Fly mode"),
                    ("Right drag", "Pan the orbit camera"),
                    ("Wheel", "Zoom; adjust movement speed in Fly mode"),
                    ("Alt + wheel", "Adjust field of view"),
                    ("W / A / S / D", "Move in Fly mode (viewport focus)"),
                    ("Q / E", "Roll in Fly mode (viewport focus)"),
                    ("Camera Reset", "Restore the camera view"),
                ],
            ),
            (
                "Editing",
                [
                    ("Edit tab", "Enable actor editing, then drag the selected actor gizmo"),
                    ("Reset Selected / All", "Discard transient actor pose edits"),
                    (
                        "Escape (Scenario Builder)",
                        "Cancel the current transform or waypoint session",
                    ),
                    ("Enter (waypoints)", "Finish the waypoint session"),
                    ("Backspace (waypoints)", "Remove the last waypoint"),
                    ("Ctrl+S (Scenario Builder)", "Save the authored scenario"),
                    ("Ctrl+Alt+S", "Save a visualization workspace snapshot"),
                ],
            ),
        ):
            page = QTableWidget(len(rows), 2)
            page.setHorizontalHeaderLabels(["Control", "Action"])
            page.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            page.horizontalHeader().setStretchLastSection(True)
            page.verticalHeader().hide()
            page.setEditTriggers(QTableWidget.NoEditTriggers)
            for row, (key, description) in enumerate(rows):
                page.setItem(row, 0, QTableWidgetItem(key))
                page.setItem(row, 1, QTableWidgetItem(description))
            page.resizeRowsToContents()
            tabs.addTab(page, title)
        layout.addWidget(tabs)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
