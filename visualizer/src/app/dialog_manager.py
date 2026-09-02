"""Qt dialog boundary for visualizer workflows.

``DialogManager`` centralizes modal file, folder, color, and message-box calls
so services and controllers can request user input through one wrapper instead
of importing Qt dialog widgets directly.
"""

from pathlib import Path
from typing import Optional

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QFileDialog,
    QMessageBox,
    QWidget,
)


class DialogManager:
    """Modal Qt dialogs owned by the main visualizer window."""

    def __init__(self, parent_window: QWidget):
        """Store the parent window used for dialog positioning and modality."""
        self.parent = parent_window

    def select_scenario_file(self, default_dir: Optional[str] = None) -> Optional[Path]:
        """Open the scenario picker and return a YAML file or folder path.

        Scenario loading accepts both direct ``scenario.yaml`` selection and
        selecting the scenario directory. The file dialog is offered first
        because it is the more explicit path; cancellation falls through to the
        folder picker instead of aborting the workflow immediately.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent,
            "Open Scenario",
            default_dir or "",
            "Scenario Files (*.yaml *.yml);;All Files (*.*)",
        )

        if file_path:
            return Path(file_path)

        folder_path = QFileDialog.getExistingDirectory(
            self.parent,
            "Open Scenario Folder",
            default_dir or "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )

        if folder_path:
            return Path(folder_path)

        return None

    def select_save_file(
        self,
        title: str = "Save File",
        default_name: str = "",
        filter_str: str = "All Files (*.*)",
    ) -> Optional[str]:
        """Open a save dialog and return the selected path, if any."""
        file_path, _ = QFileDialog.getSaveFileName(
            self.parent,
            title,
            default_name,
            filter_str,
        )

        return file_path if file_path else None

    def pick_color(
        self,
        current_color: Optional[QColor] = None,
        title: str = "Pick Color",
    ) -> Optional[QColor]:
        """Open a color picker seeded with the current visual color."""
        if current_color is None:
            current_color = QColor(255, 255, 255)

        color = QColorDialog.getColor(current_color, self.parent, title)

        if color.isValid():
            return color

        return None

    def show_info(self, title: str, message: str) -> None:
        """Show an information message box."""
        QMessageBox.information(self.parent, title, message)

    def show_warning(self, title: str, message: str) -> None:
        """Show a warning message box."""
        QMessageBox.warning(self.parent, title, message)

    def show_error(self, title: str, message: str) -> None:
        """Show an error message box."""
        QMessageBox.critical(self.parent, title, message)
