"""Browse and reuse saved environments without replacing scenario actors."""

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from .environment_library import EnvironmentLibrary
from .persistence import load_for_authoring


class EnvironmentLibraryDialog(QDialog):
    def __init__(self, workspace):
        super().__init__(workspace)
        self.workspace = workspace
        self.library = EnvironmentLibrary(project_root=workspace._project_root())
        self.entry = None
        self.setWindowTitle("Environment library")
        self.resize(690, 520)
        layout = QVBoxLayout(self)
        info = QLabel(
            "Reuse an environment with your current actors. Saved snapshots include geometry, road graphs and map overlays, and work offline."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search saved environments…")
        self.search.textChanged.connect(self.refresh)
        layout.addWidget(self.search)
        self.items = QListWidget()
        self.items.setIconSize(QSize(150, 95))
        self.items.itemSelectionChanged.connect(self._select)
        layout.addWidget(self.items, 1)
        self.detail = QLabel()
        self.detail.setWordWrap(True)
        self.detail.setTextFormat(Qt.PlainText)
        layout.addWidget(self.detail)
        actions = QHBoxLayout()
        save = QPushButton("Save current environment…")
        save.clicked.connect(self.save_current)
        save.setEnabled(workspace.document is not None and workspace._compilation_is_current())
        import_button = QPushButton("Import from scenario…")
        import_button.clicked.connect(self.import_scenario)
        actions.addWidget(save)
        actions.addWidget(import_button)
        layout.addLayout(actions)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.use = buttons.addButton("Use environment", QDialogButtonBox.AcceptRole)
        self.use.setEnabled(False)
        self.use.clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.refresh()

    def refresh(self, *_):
        self.items.clear()
        for entry in self.library.entries():
            if self.search.text().casefold() not in entry.name.casefold():
                continue
            meta = entry.metadata
            item = QListWidgetItem(
                f"{entry.name}\n{meta['scene']['source']} · saved {meta['saved_utc'][:10]}"
            )
            item.setData(Qt.UserRole, entry)
            thumbnail = entry.root / "preview.png"
            if thumbnail.is_file():
                item.setIcon(QIcon(str(thumbnail)))
            self.items.addItem(item)
        if not self.items.count():
            self.detail.setText(
                "No saved environments. Save the current one or import an existing scenario."
            )

    def _select(self):
        item = self.items.currentItem()
        self.entry = item.data(Qt.UserRole) if item is not None else None
        self.use.setEnabled(False)
        if self.entry is None:
            return
        try:
            self.library.dependencies(self.entry)
            meta = self.entry.metadata
            text = f"{len(meta.get('files', []))} saved input files"
            if meta.get("center"):
                text += f" · centre {meta['center'][0]:.5f}, {meta['center'][1]:.5f}"
            if meta.get("bbox"):
                import math

                s, w, n, e = meta["bbox"]
                text += f" · approximately {(e-w)*111320*math.cos(math.radians((n+s)/2)):.0f} × {(n-s)*111320:.0f} m"
            self.detail.setText(text + "\nYour actors and sensing settings will be preserved.")
            self.use.setEnabled(True)
        except (OSError, ValueError, TypeError) as exc:
            self.detail.setText(f"This environment is unavailable: {exc}")

    def save_current(self):
        name, accepted = QInputDialog.getText(self, "Save environment", "Name")
        if not accepted:
            return
        ws = self.workspace
        try:
            entry = self.library.save(
                ws.document.scenario,
                ws._current_scenario_directory(),
                name,
                resolved_scene=ws.compilation.resolved_scene_path,
            )
            renderer = getattr(getattr(ws.viewport, "port", None), "renderer", None)
            capture = getattr(renderer, "export_screenshot", None)
            if callable(capture):
                capture(str(entry.root / "preview.png"))
            self.refresh()
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Could not save environment", str(exc))

    def import_scenario(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import environment from scenario", "", "Scenario (*.yaml *.yml)"
        )
        if not path:
            return
        from pathlib import Path

        try:
            result = load_for_authoring(path)
            if result.document is None or result.document.scenario.scene is None:
                raise ValueError("This scenario has no supported environment.")
            root = Path(path).parent
            scenario = result.document.scenario
            resolved = (
                root / scenario.scene.id if scenario.scene.source in {"local", "osm"} else None
            )
            self.library.save(scenario, root, root.name, resolved_scene=resolved)
            self.refresh()
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Could not import environment", str(exc))
