"""Keyboard shortcut cheat-sheet dialog."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..app.shortcuts import shortcut_rows

SHORTCUTS: list[tuple[str, str]] = shortcut_rows()


class HelpDialog(QDialog):
    """Modal dialog displaying a two-column table of keyboard shortcuts."""

    def __init__(self, parent=None) -> None:
        """Build the shortcut table from the shared shortcut registry."""
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumSize(420, 340)
        flags = self.windowFlags()
        flags |= Qt.WindowSystemMenuHint | Qt.WindowCloseButtonHint
        flags &= ~Qt.WindowContextHelpButtonHint
        self.setWindowFlags(flags)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        table = QTableWidget(len(SHORTCUTS), 2)
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

        layout.addWidget(table)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
