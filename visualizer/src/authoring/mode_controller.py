"""Workspace mode switching while retaining normal and authoring sessions."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog, QMessageBox

from ..app.renderer_lifecycle import stop_renderer_session
from ..app.scenario_workflow import cleanup_previous_scene
from .document import ScenarioDocument
from .persistence import (
    AuthoringLoadResult,
    ScenarioSaveError,
    create_scenario_copy,
    load_for_authoring,
    save_document,
)
from .undo import QtUndoStackAdapter
from .workspace import ScenarioAuthoringWorkspace


class WorkspaceMode(str, Enum):
    """Top-level visualizer workspace mode."""

    VISUALIZATION = "visualization"
    AUTHORING = "authoring"


class WorkspaceModeController:
    """Own full renderer-host teardown and mode-local state retention."""

    def __init__(
        self,
        visualizer: Any,
        *,
        workspace_factory: Callable[..., ScenarioAuthoringWorkspace] | None = None,
    ) -> None:
        self.visualizer = visualizer
        self.workspace_factory = workspace_factory or ScenarioAuthoringWorkspace
        self.mode = WorkspaceMode.VISUALIZATION
        self.workspace: ScenarioAuthoringWorkspace | None = None
        self.authoring_document: ScenarioDocument | None = None
        self._read_only_result: AuthoringLoadResult | None = None
        self._normal_widget: Any = None
        self._normal_scenario_path: str | None = None
        self._normal_renderer_was_initialized = False

    def new_document(self) -> bool:
        """Create and enter an empty document after any replacement prompt."""

        if not self._can_replace_document():
            return False
        entered = self.enter_authoring(ScenarioDocument.new(undo_stack=QtUndoStackAdapter()))
        self._refresh_actions()
        return entered

    def resume_document(self) -> bool:
        """Return to the preserved in-memory draft and its existing undo stack."""

        if self.authoring_document is None:
            return False
        entered = self.enter_authoring(self.authoring_document)
        self._refresh_actions()
        return entered

    def open_document(
        self,
        source: Path | str,
    ) -> bool:
        """Open canonical compatible YAML in place or present it read-only."""

        if not self._can_replace_document():
            return False
        try:
            undo_stack = QtUndoStackAdapter()
            result = load_for_authoring(source, undo_stack=undo_stack)
            if result.editable and result.document is not None:
                return self.enter_authoring(result.document)
            self._read_only_result = result
            self.authoring_document = None
            self.enter_authoring(None)
            if self.workspace is not None:
                self.workspace.show_read_only_import(result)
            return True
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.critical(
                self.visualizer,
                "Open for Authoring Failed",
                str(exc),
            )
            return False

    def create_copy_for_authoring(
        self,
        source: Path | str,
        *,
        destination: Path | str | None = None,
    ) -> bool:
        """Persist a semantic copy of one canonical scenario and edit it."""

        if not self._can_replace_document():
            return False
        source_path = Path(source).expanduser().resolve()
        selected_destination = destination
        if selected_destination is None:
            selected = QFileDialog.getExistingDirectory(
                self.visualizer,
                "Copy Current Scenario and Edit",
                str(source_path.parent.parent),
            )
            if not selected:
                return False
            selected_destination = selected
        try:
            document = create_scenario_copy(
                source_path,
                selected_destination,
                undo_stack=QtUndoStackAdapter(),
            )
            saved = save_document(document)
            if not self.enter_authoring(document):
                return False
            # The copy is now the active branch. Returning from Builder should
            # open it, even though its generated outputs remain absent until
            # the user generates them.
            self._normal_scenario_path = str(saved.parent)
            if self.workspace is not None:
                self.workspace.append_generation_log(
                    f"Created editable scenario copy at {saved.parent}. "
                    "Generated outputs were not copied."
                )
            return True
        except ScenarioSaveError as exc:
            QMessageBox.warning(
                self.visualizer,
                "Scenario Copy Has Problems",
                str(exc),
            )
            return False
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.critical(
                self.visualizer,
                "Create Scenario Copy Failed",
                str(exc),
            )
            return False

    def enter_authoring(self, document: ScenarioDocument | None = None) -> bool:
        """Switch to an embedded pygfx authoring host without reparenting it."""

        if document is not None:
            self.authoring_document = document
            self._read_only_result = None
        elif self.authoring_document is None and self._read_only_result is None:
            self.authoring_document = ScenarioDocument.new(undo_stack=QtUndoStackAdapter())

        if self.mode is WorkspaceMode.AUTHORING and self.workspace is not None:
            if document is not None:
                self.workspace.bind_document(document)
            self._bind_generation_workspace()
            self._refresh_actions()
            self._set_workspace_status("Scenario Builder ready")
            return True

        self._normal_scenario_path = getattr(self.visualizer, "current_scenario_path", None)
        self._normal_renderer_was_initialized = bool(
            getattr(self.visualizer, "vis_initialized", False)
        )
        if self._normal_scenario_path is not None:
            # Returning to visualization already opens the scenario again from
            # this retained path. Retire its readers now so an authoring
            # generation can atomically replace the scenario's frame set on
            # platforms that prohibit renaming directories with open files.
            cleanup_previous_scene(self.visualizer)
        if self._normal_renderer_was_initialized:
            # Close the native canvas while its entire Qt ancestry is still
            # attached. The persistent stack is switched only afterward.
            stop_renderer_session(self.visualizer)

        workspace_stack = getattr(self.visualizer, "_workspace_stack", None)
        if workspace_stack is None:
            self._normal_widget = self.visualizer.takeCentralWidget()
        workspace_parent = workspace_stack or self.visualizer
        self.workspace = self.workspace_factory(
            self.visualizer,
            self.authoring_document or ScenarioDocument.new(undo_stack=QtUndoStackAdapter()),
            workspace_parent,
        )
        self.workspace.save_requested.connect(self.save)
        self.workspace.save_as_requested.connect(self.save_as)
        self.workspace.leave_authoring_requested.connect(self.request_leave_authoring)
        self.workspace.preview_generated_requested.connect(self.preview_generated_result)
        self.workspace.generate_requested.connect(self._start_generation)
        self.workspace.cancel_generation_requested.connect(self._cancel_generation)
        self.workspace.title_changed.connect(self._update_window_title)
        self.workspace.dirty_changed.connect(lambda _dirty: self._refresh_actions())
        if workspace_stack is not None:
            workspace_stack.addWidget(self.workspace)
            workspace_stack.setCurrentWidget(self.workspace)
        else:  # Lightweight test doubles expose only a central widget.
            self.visualizer.setCentralWidget(self.workspace)
        self.mode = WorkspaceMode.AUTHORING
        if self._read_only_result is not None:
            self.workspace.show_read_only_import(self._read_only_result)
        self._bind_generation_workspace()
        self._update_window_title(self.workspace.document_label.text())
        self._refresh_actions()
        self._set_workspace_status("Scenario Builder ready")
        return True

    def leave_authoring(self, *, restore_visualization: bool = True) -> bool:
        """Tear down the embedded renderer and restore the normal Qt session."""

        if self.mode is not WorkspaceMode.AUTHORING:
            return True
        workspace = self.workspace
        if workspace is not None:
            self.authoring_document = workspace.document or self.authoring_document
            workspace.close_workspace()
            workspace_stack = getattr(self.visualizer, "_workspace_stack", None)
            if workspace_stack is not None:
                normal_workspace = getattr(self.visualizer, "_visualization_workspace", None)
                if normal_workspace is not None:
                    workspace_stack.setCurrentWidget(normal_workspace)
                workspace_stack.removeWidget(workspace)
                workspace.deleteLater()
            else:
                detached = self.visualizer.takeCentralWidget()
                if detached is not None:
                    detached.deleteLater()
                if self._normal_widget is not None:
                    self.visualizer.setCentralWidget(self._normal_widget)
                    self._normal_widget = None
        self.workspace = None
        self.mode = WorkspaceMode.VISUALIZATION
        self._update_window_title("")
        self._refresh_actions()
        self._set_workspace_status("Ready")

        if (
            restore_visualization
            and self._normal_renderer_was_initialized
            and self._normal_scenario_path
        ):
            path = self._normal_scenario_path
            QTimer.singleShot(0, lambda: self.visualizer.open_scenario(path))
        return True

    def _set_workspace_status(self, message: str) -> None:
        """Replace status left by the workspace that was just retired."""

        label = getattr(self.visualizer, "status_scenario_label", None)
        if label is not None:
            label.setToolTip("")
        ui_controller = getattr(self.visualizer, "ui_controller", None)
        telemetry = getattr(ui_controller, "_telemetry_ctrl", None)
        if telemetry is not None:
            telemetry._scenario_summary_text = message
        set_status = getattr(self.visualizer, "_set_status_message", None)
        if callable(set_status):
            set_status(message)

    def prepare_normal_scenario_open(self) -> bool:
        """Leave authoring before the normal scenario workflow mutates app state.

        The normal renderer host was torn down when authoring started, so every
        normal scenario entry point must restore that host before loading.  The
        draft and its undo stack remain owned by this controller for Resume
        Authoring Draft.  Suppressing visualization restoration here also
        prevents the retained visualization scenario from racing the requested
        one.
        """

        if self.mode is not WorkspaceMode.AUTHORING:
            return True
        if not self._can_switch_to_visualization():
            return False
        return self.leave_authoring(restore_visualization=False)

    def request_leave_authoring(self) -> bool:
        """Handle an explicit Return to Visualization request through switch policy."""

        if self.mode is not WorkspaceMode.AUTHORING:
            return True
        if not self._can_switch_to_visualization():
            return False
        return self.leave_authoring()

    def save(self) -> bool:
        """Validate and atomically save the active document to scenario.yaml."""

        workspace = self.workspace
        if workspace is not None and workspace.commit_pending_edits() is False:
            return False
        document = workspace.document if workspace is not None else self.authoring_document
        return self._save_document(document, destination=None, choose_if_missing=True)

    def save_as(self) -> bool:
        """Choose a directory and save only its canonical ``scenario.yaml``."""

        workspace = self.workspace
        if workspace is not None and workspace.commit_pending_edits() is False:
            return False
        document = workspace.document if workspace is not None else self.authoring_document
        if document is None or document.read_only:
            return False
        selected = QFileDialog.getExistingDirectory(
            self.visualizer,
            "Save Scenario As Directory",
            str(document.path.parent if document.path is not None else Path.cwd()),
        )
        if not selected:
            return False
        return self._save_document(document, destination=selected, choose_if_missing=False)

    def _save_document(
        self,
        document: ScenarioDocument | None,
        *,
        destination: str | Path | None,
        choose_if_missing: bool,
    ) -> bool:
        """Shared validated save implementation for Save and Save As."""

        workspace = self.workspace
        if document is None or document.read_only:
            return False
        generation = getattr(self.visualizer, "authoring_generation_controller", None)
        if bool(getattr(generation, "running", False)):
            if workspace is not None:
                workspace.append_generation_log(
                    "Save is disabled while generation owns snapshot/promotion paths."
                )
            return False
        if destination is None and document.path is None and choose_if_missing:
            selected = QFileDialog.getExistingDirectory(
                self.visualizer,
                "Save Scenario Directory",
                str(Path.cwd()),
            )
            if not selected:
                return False
            destination = selected
        try:
            saved = save_document(
                document,
                destination,
                compiler=getattr(workspace, "compiler", None),
                compile_lock=getattr(workspace, "compilation_lock", None),
            )
        except ScenarioSaveError as exc:
            if workspace is not None:
                workspace.refresh_now()
            QMessageBox.warning(self.visualizer, "Scenario Has Problems", str(exc))
            return False
        except (OSError, ValueError, PermissionError) as exc:
            QMessageBox.critical(self.visualizer, "Save Failed", str(exc))
            return False
        if workspace is not None:
            workspace.append_generation_log(f"Saved {saved}")
        self._refresh_actions()
        return True

    def preview_generated_result(self, scenario_path: Any = None) -> None:
        """Switch to playback while retaining the authoring document and undo stack."""

        document = self.authoring_document
        generation = getattr(self.visualizer, "authoring_generation_controller", None)
        if scenario_path is None:
            scenario_path = getattr(generation, "preview_path", None)
        if scenario_path is None and document is not None and document.path is not None:
            scenario_path = document.path.parent
        if scenario_path is None:
            if self.workspace is not None:
                self.workspace.append_generation_log(
                    "Generated preview is unavailable because no scenario path was produced."
                )
            return
        self.leave_authoring(restore_visualization=False)
        QTimer.singleShot(0, lambda: self._open_preview_or_resume(str(scenario_path)))

    def _open_preview_or_resume(self, scenario_path: str) -> None:
        """Open generated playback or restore the preserved draft on failure."""

        try:
            outcome = self.visualizer.open_scenario(scenario_path)
        except Exception as exc:
            outcome = None
            failure_message = f"Generated preview could not be opened: {exc}"
        else:
            if bool(getattr(outcome, "succeeded", False)):
                return
            failure_message = (
                getattr(outcome, "message", None) or "Generated preview could not be opened."
            )
        document = self.authoring_document
        if document is None:
            return
        if self.enter_authoring(document) and self.workspace is not None:
            self.workspace.append_generation_log(failure_message)

    def request_application_close(self) -> bool:
        """Prompt only when application close could discard a dirty draft."""

        generation = getattr(self.visualizer, "authoring_generation_controller", None)
        if bool(getattr(generation, "running", False)):
            choice = QMessageBox.question(
                self.visualizer,
                "Generation Running",
                "Cancel generation and close ORCHAV?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if choice != QMessageBox.Yes:
                return False
            generation.cancel()
        if self.mode is WorkspaceMode.AUTHORING and self.workspace is not None:
            return self.workspace.confirm_replace()
        if self.authoring_document is not None and self.authoring_document.dirty:
            choice = QMessageBox.question(
                self.visualizer,
                "Unsaved Scenario",
                "Discard the preserved unsaved Scenario Builder draft and close ORCHAV?",
                QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            return choice == QMessageBox.Discard
        return True

    def close(self) -> None:
        """Release an active authoring viewport during application shutdown."""

        if self.workspace is not None:
            self.workspace.close_workspace()
            workspace_stack = getattr(self.visualizer, "_workspace_stack", None)
            if workspace_stack is not None:
                workspace_stack.removeWidget(self.workspace)
            self.workspace = None

    def _can_replace_document(self) -> bool:
        generation = getattr(self.visualizer, "authoring_generation_controller", None)
        if bool(getattr(generation, "running", False)):
            if self.workspace is not None:
                self.workspace.append_generation_log(
                    "Finish or cancel generation before replacing the authoring document."
                )
            return False
        if self.workspace is not None:
            return self.workspace.confirm_replace()
        document = self.authoring_document
        if document is None or not document.dirty:
            return True
        choice = QMessageBox.question(
            self.visualizer,
            "Unsaved Scenario",
            "Discard the preserved unsaved Scenario Builder draft?",
            QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return choice == QMessageBox.Discard

    def _can_switch_to_visualization(self) -> bool:
        """Confirm a dirty mode switch without implying that the draft is discarded."""

        generation = getattr(self.visualizer, "authoring_generation_controller", None)
        if bool(getattr(generation, "running", False)):
            if self.workspace is not None:
                self.workspace.append_generation_log(
                    "Finish or cancel generation before switching to visualization."
                )
            return False
        if self.workspace is not None and self.workspace.commit_pending_edits() is False:
            return False
        document = (
            self.workspace.document if self.workspace is not None else self.authoring_document
        )
        if document is None or not document.dirty:
            return True
        choice = QMessageBox.question(
            self.visualizer,
            "Switch to Visualization",
            "Switch to visualization? The Scenario Builder draft and undo history "
            "will be preserved.",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return choice == QMessageBox.Yes

    def _update_window_title(self, document_title: str) -> None:
        title = "ORCHAV"
        if self.mode is WorkspaceMode.AUTHORING:
            title += " — Scenario Builder"
            if document_title:
                title += f" — {document_title}"
        self.visualizer.setWindowTitle(title)

    def _start_generation(self) -> None:
        controller = getattr(self.visualizer, "authoring_generation_controller", None)
        start = getattr(controller, "start", None)
        if callable(start) and self.authoring_document is not None:
            if self.workspace is not None and self.workspace.commit_pending_edits() is False:
                return
            start(self.authoring_document)

    def _cancel_generation(self) -> None:
        controller = getattr(self.visualizer, "authoring_generation_controller", None)
        cancel = getattr(controller, "cancel", None)
        if callable(cancel):
            cancel()

    def _refresh_actions(self) -> None:
        refresh = getattr(self.visualizer, "_refresh_authoring_actions", None)
        if callable(refresh):
            refresh()

    def _bind_generation_workspace(self) -> None:
        """Restore controller-owned generation state after workspace construction."""

        generation = getattr(self.visualizer, "authoring_generation_controller", None)
        bind_workspace = getattr(generation, "bind_workspace", None)
        if self.workspace is not None and callable(bind_workspace):
            bind_workspace(self.workspace)
