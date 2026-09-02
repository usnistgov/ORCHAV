"""Qt bridge for background :class:`GenerationJob` callbacks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import yaml
from PySide6.QtCore import QObject, Signal, Slot

from .document import DocumentEvent, ScenarioDocument
from .generation import (
    GenerationJob,
    GenerationProgress,
    GenerationResult,
    GenerationState,
)


@dataclass(frozen=True, slots=True)
class GenerationLogEntry:
    """One controller-owned generation log line with its original channel."""

    stream_name: str
    line: str

    @property
    def display_text(self) -> str:
        """Return the text shown in the workspace generation log."""

        if self.stream_name == "status":
            return self.line
        prefix = "stderr" if self.stream_name == "stderr" else "stdout"
        return f"[{prefix}] {self.line}"


class QtGenerationController(QObject):
    """Launch, observe, and cancel one authoring generation job at a time.

    ``GenerationJob`` callbacks originate on reader/waiter threads. They emit
    private Qt signals here, ensuring every workspace mutation runs on the GUI
    thread through queued signal delivery.
    """

    progress_changed = Signal(object)
    log_received = Signal(str, str)
    finished = Signal(object)

    _worker_progress = Signal(object)
    _worker_log = Signal(str, str)
    _worker_finished = Signal(object)

    def __init__(
        self,
        *,
        save_callback: Callable[[], bool],
        workspace_provider: Callable[[], Any | None],
        job_factory: Callable[..., GenerationJob] = GenerationJob,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._save_callback = save_callback
        self._workspace_provider = workspace_provider
        self._job_factory = job_factory
        self._job: GenerationJob | None = None
        self._document: ScenarioDocument | None = None
        self._unsubscribe_document: Callable[[], None] | None = None
        self._last_result: GenerationResult | None = None
        self._latest_progress: GenerationProgress | None = None
        self._log_entries: list[GenerationLogEntry] = []
        self._launched_revision: int | None = None
        self._generation_serial = 0
        self._worker_progress.connect(self._apply_progress)
        self._worker_log.connect(self._apply_log)
        self._worker_finished.connect(self._apply_finished)

    @property
    def job(self) -> GenerationJob | None:
        return self._job

    @property
    def running(self) -> bool:
        return self._job is not None and self._job.state is GenerationState.RUNNING

    @property
    def last_result(self) -> GenerationResult | None:
        return self._last_result

    @property
    def latest_progress(self) -> GenerationProgress | None:
        """Return the latest structured progress for workspace reconstruction."""

        return self._latest_progress

    @property
    def log_entries(self) -> tuple[GenerationLogEntry, ...]:
        """Return the current generation's ordered status/stdout/stderr log."""

        return tuple(self._log_entries)

    @property
    def preview_path(self) -> Path | None:
        result = self._last_result
        return (
            result.paths.scenario_yaml.parent if result is not None and result.succeeded else None
        )

    def start(self, document: ScenarioDocument) -> bool:
        """Save the current draft revision, then launch its immutable snapshot."""

        if self.running:
            self._append_status("Generation is already running.")
            return False
        # Always cross the canonical save boundary, including for a document
        # that currently appears clean. Reopened files may have become invalid
        # because an asset disappeared or the YAML changed on disk; generation
        # must never bypass the same schema/factory/preparation checks as Save.
        if not self._save_callback():
            self._append_status("Generation not started because the draft was not saved.")
            return False
        if document.path is None:
            self._append_status("Generation requires a saved scenario.yaml.")
            return False

        launched_revision = document.revision
        if self._unsubscribe_document is not None:
            self._unsubscribe_document()
        self._document = document
        self._unsubscribe_document = document.subscribe(self._document_changed)
        self._generation_serial += 1
        self._launched_revision = launched_revision
        self._latest_progress = None
        self._log_entries.clear()
        self._last_result = None
        try:
            job = self._job_factory(
                document.path,
                launched_revision,
                revision_provider=lambda: document.revision,
                on_progress=self._worker_progress.emit,
                on_log=self._worker_log.emit,
                on_finished=self._worker_finished.emit,
            )
        except (OSError, ValueError, yaml.YAMLError) as exc:
            self._append_status(f"Generation not started: {exc}")
            self._refresh_actions()
            return False
        self._job = job
        workspace = self._workspace_provider()
        if workspace is not None:
            reset_state = getattr(workspace, "reset_generation_state", None)
            if callable(reset_state):
                reset_state()
            workspace.set_generation_running(True, launched_revision=launched_revision)
        self._append_status(
            f"Starting generation from saved document revision {launched_revision}."
        )
        self._refresh_actions()
        job.start()
        return job.state in {GenerationState.RUNNING, GenerationState.SUCCEEDED}

    def cancel(self) -> bool:
        """Request cooperative termination of the active generator process."""

        job = self._job
        if job is None or not job.cancel():
            return False
        self._append_status("Cancellation requested...")
        return True

    def shutdown(self) -> None:
        """Cancel and join an active subprocess before Qt is destroyed."""

        if self.running:
            self.cancel()
        job = self._job
        wait = getattr(job, "wait", None)
        if job is not None and job.state is GenerationState.RUNNING and callable(wait):
            # GenerationJob escalates terminate to kill after its bounded grace
            # interval. Waiting here prevents Qt teardown from orphaning the
            # waiter thread or its generator child process.
            wait()
        if self._unsubscribe_document is not None:
            self._unsubscribe_document()
            self._unsubscribe_document = None

    def bind_workspace(self, workspace: Any) -> None:
        """Replay controller-owned generation state into one active workspace."""

        if workspace is None:
            return
        if self._document is None or getattr(workspace, "document", None) is not self._document:
            reset_state = getattr(workspace, "reset_generation_state", None)
            if callable(reset_state):
                reset_state()
            return

        if self.running:
            workspace.set_generation_running(
                True,
                launched_revision=self._launched_revision,
            )
            if self._latest_progress is not None:
                workspace.set_generation_progress(self._latest_progress)
        elif self._last_result is not None:
            if self._latest_progress is not None:
                workspace.set_generation_progress(self._latest_progress)
            workspace.set_generation_result(self._last_result)
        self._replay_logs(workspace)

    @Slot(object)
    def _apply_progress(self, progress: GenerationProgress) -> None:
        self._latest_progress = progress
        workspace = self._workspace_provider()
        if workspace is not None:
            workspace.set_generation_progress(progress)
        self.progress_changed.emit(progress)

    @Slot(str, str)
    def _apply_log(self, stream_name: str, line: str) -> None:
        self._record_log(stream_name, line)
        self.log_received.emit(stream_name, line)

    @Slot(object)
    def _apply_finished(self, result: GenerationResult) -> None:
        self._last_result = result
        if result.progress is not None:
            self._latest_progress = result.progress
        workspace = self._workspace_provider()
        if workspace is not None:
            workspace.set_generation_result(result)
        if result.stale:
            self._append_status(
                "Generated result corresponds to an older draft revision: "
                "it was launched from revision "
                f"{result.launched_revision}, while the draft is revision "
                f"{result.current_revision}."
            )
        self._refresh_actions()
        self.finished.emit(result)

    def _document_changed(self, event: DocumentEvent) -> None:
        """Keep a completed result's stale label current after later edits."""

        result = self._last_result
        if result is None:
            return
        stale = int(event.revision) != result.launched_revision
        if stale == result.stale and int(event.revision) == result.current_revision:
            return
        became_stale = stale and not result.stale
        updated = replace(
            result,
            current_revision=int(event.revision),
            stale=stale,
        )
        self._last_result = updated
        workspace = self._workspace_provider()
        if workspace is not None:
            workspace.set_generation_result(updated)
        if became_stale:
            self._append_status(
                "Generated result now corresponds to an older draft revision: "
                f"it was launched from revision {updated.launched_revision}, while "
                f"the draft is revision {updated.current_revision}."
            )

    def _append_status(self, message: str) -> None:
        self._record_log("status", message)

    def _record_log(self, stream_name: str, line: str) -> None:
        """Buffer one ordered log entry and deliver any unseen entries."""

        self._log_entries.append(GenerationLogEntry(str(stream_name), str(line)))
        workspace = self._workspace_provider()
        if workspace is not None:
            self._replay_logs(workspace)

    def _replay_logs(self, workspace: Any) -> None:
        """Append only entries this workspace has not already received."""

        token = (id(self), self._generation_serial)
        previous_token = getattr(workspace, "_orchav_generation_replay_token", None)
        offset = (
            int(getattr(workspace, "_orchav_generation_replay_offset", 0))
            if previous_token == token
            else 0
        )
        offset = max(0, min(offset, len(self._log_entries)))
        for entry in self._log_entries[offset:]:
            workspace.append_generation_log(entry.display_text)
        workspace._orchav_generation_replay_token = token
        workspace._orchav_generation_replay_offset = len(self._log_entries)

    def _refresh_actions(self) -> None:
        refresh = getattr(self.parent(), "_refresh_authoring_actions", None)
        if callable(refresh):
            refresh()


__all__ = ["GenerationLogEntry", "QtGenerationController"]
