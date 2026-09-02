"""Menu bar setup, recent-file actions, and scenario-open routing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QLabel, QMessageBox, QProgressBar

from ..app.shortcuts import shortcut
from ..app.theme import ThemeMode, get_theme_manager
from ..authoring.feature import scenario_builder_enabled
from ..renderers.protocol import renderer_capabilities
from ..scene.defaults import DEFAULT_SCENE_BACKGROUND_COLOR

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer

logger = logging.getLogger(__name__)


class MenuController:
    """Build and refresh menu actions owned by the visualizer window.

    The controller wires QAction callbacks to their owning controllers and
    services. It does not make the root window a command-routing facade.
    """

    def __init__(self, parent: Any) -> None:
        """Initialize menu state before the main window creates QActions."""
        self._parent = parent
        self._scenario_loader: Any = None
        self._menu_parent: Optional[Any] = None
        self._recent_menu: Optional[Any] = None
        self._recent_sessions_menu: Optional[Any] = None

    @property
    def visualizer(self) -> OrchavVisualizer:
        """Shortcut to the parent's visualizer instance."""
        return self._parent.visualizer

    def setup_menus(self, parent: Any, metrics_available: bool) -> None:
        """Create menus and status-bar widgets on the main window.

        Args:
            parent: The QMainWindow that owns the menu bar.
            metrics_available: Whether the metrics window action should be added.
        """
        self._menu_parent = parent
        menubar = parent.menuBar()
        file_menu = menubar.addMenu("&File")
        renderer = getattr(parent, "renderer", None)
        authoring_available = bool(
            scenario_builder_enabled()
            and renderer is not None
            and renderer_capabilities(renderer).scenario_authoring
        )
        open_act = QAction("Open Scenario...", parent)
        open_act.triggered.connect(self.open_scenario_dialog)
        file_menu.addAction(open_act)

        parent.recent_menu = file_menu.addMenu("Open Recent")
        self._recent_menu = parent.recent_menu
        self.update_recent_menu()

        file_menu.addSeparator()
        session_menu = file_menu.addMenu("&Workspace")

        save_session_act = QAction("&Save Workspace Snapshot...", parent)
        save_session_act.setShortcut(shortcut("save_session").key_sequence())
        save_session_act.triggered.connect(parent.save_session_dialog)
        session_menu.addAction(save_session_act)

        load_session_act = QAction("&Open Workspace Snapshot...", parent)
        load_session_act.setShortcut(shortcut("load_session").key_sequence())
        load_session_act.triggered.connect(parent.load_session_dialog)
        session_menu.addAction(load_session_act)

        session_menu.addSeparator()
        parent.recent_sessions_menu = session_menu.addMenu("Recent Workspace Snapshots")
        self._recent_sessions_menu = parent.recent_sessions_menu
        self.update_recent_sessions_menu()

        if authoring_available:
            builder_menu = menubar.addMenu("&Scenario Builder")
            parent.scenario_builder_menu = builder_menu

            builder_menu.addSection("Create")
            new_authoring_act = QAction("New Scenario", parent)
            new_authoring_act.triggered.connect(parent.new_authoring_scenario)
            builder_menu.addAction(new_authoring_act)
            parent.new_authoring_scenario_action = new_authoring_act

            copy_current_act = QAction("Copy Current Scenario and Edit...", parent)
            copy_current_act.triggered.connect(parent.copy_current_scenario_and_edit)
            copy_current_act.setEnabled(False)
            builder_menu.addAction(copy_current_act)
            parent.copy_current_scenario_action = copy_current_act

            builder_menu.addSection("Edit or Continue")
            open_authoring_act = QAction("Open Scenario for Authoring...", parent)
            open_authoring_act.triggered.connect(parent.open_for_authoring_dialog)
            builder_menu.addAction(open_authoring_act)
            parent.open_for_authoring_action = open_authoring_act

            edit_current_act = QAction("Edit Current Scenario", parent)
            edit_current_act.triggered.connect(parent.edit_current_scenario)
            edit_current_act.setEnabled(False)
            builder_menu.addAction(edit_current_act)
            parent.edit_current_scenario_action = edit_current_act

            resume_authoring_act = QAction("Resume Authoring Draft", parent)
            resume_authoring_act.triggered.connect(parent.resume_authoring_draft)
            resume_authoring_act.setEnabled(False)
            builder_menu.addAction(resume_authoring_act)
            parent.resume_authoring_draft_action = resume_authoring_act

            builder_menu.addSection("Save")
            save_authoring_act = QAction("Save Scenario", parent)
            save_authoring_act.setShortcut("Ctrl+S")
            save_authoring_act.triggered.connect(parent.save_authoring_scenario)
            save_authoring_act.setEnabled(False)
            builder_menu.addAction(save_authoring_act)
            parent.save_authoring_scenario_action = save_authoring_act

            save_as_authoring_act = QAction("Save Scenario As...", parent)
            save_as_authoring_act.setShortcut("Ctrl+Shift+S")
            save_as_authoring_act.triggered.connect(parent.save_authoring_scenario_as)
            save_as_authoring_act.setEnabled(False)
            builder_menu.addAction(save_as_authoring_act)
            parent.save_authoring_scenario_as_action = save_as_authoring_act

            builder_menu.addSection("Workspace")
            return_to_visualization_act = QAction("Return to Visualization", parent)
            return_to_visualization_act.triggered.connect(parent.return_to_visualization)
            return_to_visualization_act.setEnabled(False)
            builder_menu.addAction(return_to_visualization_act)
            parent.return_to_visualization_action = return_to_visualization_act

        view_menu = menubar.addMenu("&View")
        if metrics_available:
            metrics_act = QAction("&Metrics Window", parent)
            metrics_act.setShortcut(shortcut("toggle_metrics").key_sequence())
            metrics_act.triggered.connect(self._parent.toggle_metrics_window)
            view_menu.addAction(metrics_act)

        display_menu = menubar.addMenu("&Display")
        theme_menu = display_menu.addMenu("&Theme")
        theme_group = QActionGroup(parent)
        theme_group.setExclusive(True)
        theme_manager = get_theme_manager()
        for mode, label in (
            (ThemeMode.SYSTEM, "System"),
            (ThemeMode.LIGHT, "Light"),
            (ThemeMode.DARK, "Dark"),
        ):
            action = QAction(label, parent)
            action.setCheckable(True)
            action.setData(mode.value)
            action.setChecked(theme_manager.mode == mode)
            action.triggered.connect(lambda _checked=False, m=mode: theme_manager.set_mode(m))
            theme_group.addAction(action)
            theme_menu.addAction(action)
        parent.theme_action_group = theme_group

        bg_menu = display_menu.addMenu("&Background Color")
        scene_appearance = self.visualizer.scene_appearance_service
        for name, color in self._background_color_options():
            action = QAction(name, parent)
            if color is not None:
                action.triggered.connect(
                    lambda _checked=False, c=color: scene_appearance.set_background_color(c)
                )
            else:
                action.triggered.connect(scene_appearance.pick_custom_background_color)
            bg_menu.addAction(action)

        bg_menu.addSeparator()
        current_bg = QAction("&Current: Dark Gray", parent)
        current_bg.setEnabled(False)
        bg_menu.addAction(current_bg)
        parent.current_bg_action = current_bg

        reset_action = QAction("&Reset to Default (Dark Gray)", parent)
        reset_action.triggered.connect(scene_appearance.reset_to_default_background)
        bg_menu.addAction(reset_action)

        help_menu = menubar.addMenu("&Help")
        shortcuts_act = QAction("&Keyboard Shortcuts", parent)
        shortcuts_act.triggered.connect(parent._show_help_dialog)
        help_menu.addAction(shortcuts_act)

        # --- 3-zone status bar: [scenario context] [progress] [playback cadence] ---
        parent.status_scenario_label = QLabel("")
        parent.status_scenario_label.setProperty("role", "secondary")
        parent.status_scenario_label.setStyleSheet("padding-left: 8px;")
        parent.statusBar().addPermanentWidget(parent.status_scenario_label, 1)

        parent.status_progress_bar = QProgressBar()
        parent.status_progress_bar.setFixedWidth(200)
        parent.status_progress_bar.setTextVisible(True)
        parent.status_progress_bar.setVisible(False)
        parent.statusBar().addPermanentWidget(parent.status_progress_bar)

        parent.status_fps_label = QLabel("Playback updates: paused")
        parent.status_fps_label.setProperty("role", "secondary")
        parent.status_fps_label.setStyleSheet("padding-left: 8px;")
        parent.statusBar().addPermanentWidget(parent.status_fps_label)
        parent._set_status_message("Ready")

    @staticmethod
    def _background_color_options() -> List[tuple[str, Any]]:
        """Return the list of background color menu entries."""
        return [
            ("&Default (Dark Gray)", DEFAULT_SCENE_BACKGROUND_COLOR),
            ("&Black", [0.0, 0.0, 0.0]),
            ("&White", [1.0, 1.0, 1.0]),
            ("&Light Gray", [0.8, 0.8, 0.8]),
            ("&Dark Blue", [0.1, 0.1, 0.3]),
            ("&Dark Green", [0.1, 0.3, 0.1]),
            ("&Dark Red", [0.3, 0.1, 0.1]),
            ("&Custom...", None),
        ]

    def update_recent_menu(self) -> None:
        """Rebuild the recent scenario/XML-scene submenu from app state."""
        menu = self._recent_menu
        if menu is None:
            return
        menu.clear()

        recent_files: List[str] = getattr(self.visualizer, "recent_files", [])
        if not recent_files:
            action = QAction("No recent files", self._menu_parent)
            action.setEnabled(False)
            menu.addAction(action)
            return

        for idx, file_path in enumerate(recent_files):
            display_name = Path(file_path).name
            action = QAction(f"{idx+1}. {display_name}", self._menu_parent)
            action.setData(file_path)
            action.triggered.connect(lambda _, path=file_path: self.open_recent_file(path))
            menu.addAction(action)

        menu.addSeparator()
        refresh_action = QAction("Refresh Recent", self._menu_parent)
        refresh_action.triggered.connect(self.refresh_recent_files)
        menu.addAction(refresh_action)
        clear_action = QAction("Clear Recent", self._menu_parent)
        clear_action.triggered.connect(self.clear_recent_files)
        menu.addAction(clear_action)

    def update_recent_sessions_menu(self) -> None:
        """Rebuild the recent workspace submenu from snapshot metadata."""
        menu = self._recent_sessions_menu
        if menu is None:
            return
        menu.clear()

        if not hasattr(self.visualizer, "session_service"):
            action = QAction("No workspace snapshots", self._menu_parent)
            action.setEnabled(False)
            menu.addAction(action)
            return

        snapshots = self.visualizer.session_service.list_workspace_summaries(max_count=5)
        if not snapshots:
            action = QAction("No workspace snapshots", self._menu_parent)
            action.setEnabled(False)
            menu.addAction(action)
            return

        for idx, summary in enumerate(snapshots):
            scenario_name = summary.scenario_name.replace("_", " ").replace("-", " ").title()
            saved_at = summary.created_at.strftime("%b %d %H:%M")
            if summary.is_autosave:
                display_name = f"{scenario_name} — frame {summary.frame} — {saved_at} (Auto)"
            else:
                snapshot_name = summary.path.stem.replace("_", " ").replace("-", " ").title()
                display_name = (
                    f"{snapshot_name} — {scenario_name} — frame {summary.frame} — "
                    f"{saved_at} (Manual)"
                )
            action = QAction(f"{idx + 1}. {display_name}", self._menu_parent)
            action.setData(str(summary.path))
            action.setToolTip(
                f"Scenario: {summary.scenario_root}\n"
                f"Saved: {summary.created_at.isoformat(sep=' ', timespec='seconds')}\n"
                f"Snapshot: {summary.path}"
            )
            action.triggered.connect(
                lambda _, path=summary.path: self.visualizer.load_session_file(path)
            )
            menu.addAction(action)

    def register_scenario_loader(self, loader: Any) -> None:
        """Inject the scenario loader service once it is available.

        Args:
            loader: A scenario loader service with an ``open_scenario_via_dialog`` method.
        """
        self._scenario_loader = loader

    def open_scenario_dialog(self) -> None:
        """Delegate scenario dialog handling to the loader service."""
        if not self._scenario_loader:
            logger.warning("Scenario loader service not configured; ignoring dialog request")
            return
        self._scenario_loader.open_scenario_via_dialog()

    def add_recent_file(self, file_path: str) -> None:
        """Add a file path to the recent list and persist the change."""
        from ..io.config_handlers import RecentFilesHandler

        recent = RecentFilesHandler.add_recent_file(
            self.visualizer.config_file,
            file_path,
            self.visualizer.max_recent_files,
        )
        self.visualizer.recent_files = recent
        self.update_recent_menu()

    def open_recent_file(self, file_path: str) -> None:
        """Open a recent scenario directory/YAML or direct XML scene path."""
        from ..io.config_handlers import RecentFilesHandler

        path_obj = Path(file_path)
        if not path_obj.exists():
            if file_path in self.visualizer.recent_files:
                self.visualizer.recent_files.remove(file_path)
                self.update_recent_menu()
                self.visualizer.recent_files = RecentFilesHandler.save_recent_files(
                    self.visualizer.config_file, self.visualizer.recent_files
                )
            QMessageBox.warning(
                self._menu_parent,
                "File Not Found",
                f"The scenario '{file_path}' no longer exists.",
            )
            return

        if path_obj.is_file() and path_obj.suffix.lower() in {".yaml", ".yml"}:
            scenario_dir = path_obj.parent
        elif path_obj.is_dir():
            scenario_dir = path_obj
        elif path_obj.is_file() and path_obj.suffix.lower() == ".xml":
            try:
                self.visualizer.main_controller.load_scene(file_path)
            except (OSError, ValueError, RuntimeError) as exc:
                logger.error("Could not open XML scene %s: %s", file_path, exc)
                QMessageBox.critical(
                    self._menu_parent,
                    "Scene Error",
                    f"Failed to open scene:\n{exc}",
                )
                return
            self.add_recent_file(file_path)
            return
        else:
            QMessageBox.warning(
                self._menu_parent,
                "Unsupported File",
                f"The recent path is not a scenario directory, YAML file, or XML scene:\n"
                f"{file_path}",
            )
            return

        try:
            from ..io.scenario_config import find_project_root

            project_root = find_project_root()
            try:
                relative_path = scenario_dir.relative_to(project_root)
                scenario_path_str = str(relative_path)
            except ValueError:
                scenario_path_str = str(scenario_dir)
        except (ImportError, OSError, ValueError):
            scenario_path_str = str(scenario_dir)

        self.visualizer.open_scenario(scenario_path_str)

    def clear_recent_files(self) -> None:
        """Clear the recent files list."""
        from ..io.config_handlers import RecentFilesHandler

        self.visualizer.recent_files.clear()
        self.update_recent_menu()
        self.visualizer.recent_files = RecentFilesHandler.save_recent_files(
            self.visualizer.config_file, self.visualizer.recent_files
        )

    def refresh_recent_files(self) -> None:
        """Remove missing paths from the recent-files list and persist it."""
        from ..io.config_handlers import RecentFilesHandler

        original_count = len(self.visualizer.recent_files)
        self.visualizer.recent_files = [
            entry for entry in self.visualizer.recent_files if Path(entry).exists()
        ]
        removed = original_count - len(self.visualizer.recent_files)

        if removed:
            QMessageBox.information(
                self._menu_parent,
                "Recent Files Refreshed",
                f"Removed {removed} invalid file(s) from recent list.",
            )
            self.visualizer.recent_files = RecentFilesHandler.save_recent_files(
                self.visualizer.config_file, self.visualizer.recent_files
            )
        else:
            QMessageBox.information(
                self._menu_parent, "Recent Files Refreshed", "All recent files are valid."
            )
        self.update_recent_menu()
