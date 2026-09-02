"""Unit tests for DialogManager."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget

from visualizer.src.app.dialog_manager import DialogManager


@pytest.fixture
def parent_widget():
    """Create a mock parent widget for DialogManager."""
    return Mock(spec=QWidget)


@pytest.fixture
def dialog_manager(parent_widget):
    """Create a DialogManager instance for testing."""
    return DialogManager(parent_widget)


class TestDialogManager:
    """Tests for DialogManager class."""

    def test_init(self, parent_widget):
        """Test DialogManager initialization."""
        manager = DialogManager(parent_widget)
        assert manager.parent is parent_widget

    @patch("visualizer.src.app.dialog_manager.QFileDialog.getOpenFileName")
    def test_select_scenario_file_yaml(self, mock_dialog, dialog_manager):
        """Test selecting a YAML scenario file."""
        test_path = "/path/to/scenario.yaml"
        mock_dialog.return_value = (test_path, "Scenario Files (*.yaml *.yml)")

        result = dialog_manager.select_scenario_file()

        assert result == Path(test_path)
        mock_dialog.assert_called_once()

    @patch("visualizer.src.app.dialog_manager.QFileDialog.getExistingDirectory")
    @patch("visualizer.src.app.dialog_manager.QFileDialog.getOpenFileName")
    def test_select_scenario_file_folder(self, mock_open_file, mock_existing_dir, dialog_manager):
        """Test selecting a scenario folder when file dialog is cancelled."""
        test_folder = "/path/to/scenario"
        mock_open_file.return_value = ("", "")  # Cancelled
        mock_existing_dir.return_value = test_folder

        result = dialog_manager.select_scenario_file()

        assert result == Path(test_folder)
        mock_open_file.assert_called_once()
        mock_existing_dir.assert_called_once()

    @patch("visualizer.src.app.dialog_manager.QFileDialog.getExistingDirectory")
    @patch("visualizer.src.app.dialog_manager.QFileDialog.getOpenFileName")
    def test_select_scenario_file_cancelled(
        self, mock_open_file, mock_existing_dir, dialog_manager
    ):
        """Test when both file and folder dialogs are cancelled."""
        mock_open_file.return_value = ("", "")
        mock_existing_dir.return_value = ""

        result = dialog_manager.select_scenario_file()

        assert result is None

    @patch("visualizer.src.app.dialog_manager.QFileDialog.getSaveFileName")
    def test_select_save_file(self, mock_dialog, dialog_manager):
        """Test save file dialog."""
        test_path = "/path/to/output.xml"
        mock_dialog.return_value = (test_path, "XML Files (*.xml)")

        result = dialog_manager.select_save_file(
            title="Save XML",
            default_name="scene.xml",
            filter_str="XML Files (*.xml)",
        )

        assert result == test_path
        mock_dialog.assert_called_once()

    @patch("visualizer.src.app.dialog_manager.QFileDialog.getSaveFileName")
    def test_select_save_file_cancelled(self, mock_dialog, dialog_manager):
        """Test save file dialog cancellation."""
        mock_dialog.return_value = ("", "")

        result = dialog_manager.select_save_file()

        assert result is None

    @patch("visualizer.src.app.dialog_manager.QColorDialog.getColor")
    def test_pick_color(self, mock_dialog, dialog_manager):
        """Test color picker dialog."""
        test_color = QColor(255, 128, 64)
        mock_dialog.return_value = test_color

        result = dialog_manager.pick_color(
            current_color=QColor(0, 0, 0),
            title="Pick Test Color",
        )

        assert result == test_color
        mock_dialog.assert_called_once()

    @patch("visualizer.src.app.dialog_manager.QColorDialog.getColor")
    def test_pick_color_cancelled(self, mock_dialog, dialog_manager):
        """Test color picker dialog cancellation."""
        invalid_color = QColor()  # Invalid color
        mock_dialog.return_value = invalid_color

        result = dialog_manager.pick_color()

        assert result is None

    @patch("visualizer.src.app.dialog_manager.QMessageBox.information")
    def test_show_info(self, mock_dialog, dialog_manager):
        """Test information message box."""
        dialog_manager.show_info("Test Title", "Test Message")

        mock_dialog.assert_called_once_with(dialog_manager.parent, "Test Title", "Test Message")

    @patch("visualizer.src.app.dialog_manager.QMessageBox.warning")
    def test_show_warning(self, mock_dialog, dialog_manager):
        """Test warning message box."""
        dialog_manager.show_warning("Warning Title", "Warning Message")

        mock_dialog.assert_called_once_with(
            dialog_manager.parent, "Warning Title", "Warning Message"
        )

    @patch("visualizer.src.app.dialog_manager.QMessageBox.critical")
    def test_show_error(self, mock_dialog, dialog_manager):
        """Test error message box."""
        dialog_manager.show_error("Error Title", "Error Message")

        mock_dialog.assert_called_once_with(dialog_manager.parent, "Error Title", "Error Message")
