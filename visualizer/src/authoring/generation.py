"""Subprocess generation for immutable Scenario Builder revisions.

The authoring UI may continue changing its in-memory document while a run is
active. ``GenerationJob`` therefore launches the generator from a private YAML
snapshot while keeping the saved scenario root authoritative. The generator's
shared writers own the one and only frame/summary transaction; the Builder
observes progress and verifies the newly published canonical frame identity.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TextIO, cast

import yaml

from shared.frames.directory_ownership import (
    compact_uuid_token,
    preflight_windows_transaction_paths,
)
from shared.frames.manifest import FrameManifestError, load_frame_manifest
from shared.logging import get_logger
from shared.scenarios.frame_paths import DEFAULT_FRAMES_DIRECTORY
from shared.scenarios.yaml import validate_scenario_data
from shared.source_identity import (
    EXPECTED_SOURCE_IDENTITY_ENV,
    SourceIdentity,
    loaded_source_identity,
    source_bound_module_command,
)

logger = get_logger(__name__)

_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_SNAPSHOT_PREFIX = ".orchav-aq-"
_PROGRESS_SCHEMA_VERSION = 1
_PROGRESS_EVENTS = {"run_started", "step_completed", "run_completed", "run_failed"}


class GenerationState(str, Enum):
    """Lifecycle states for one immutable generation launch."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class GenerationPaths:
    """Saved input, immutable snapshot, and canonical published frame path."""

    scenario_yaml: Path
    snapshot_yaml: Path
    final_frames: Path


def _preflight_generation_paths(paths: GenerationPaths) -> None:
    """Fail early when the one Builder-owned snapshot path is too deep."""

    preflight_windows_transaction_paths([paths.snapshot_yaml])


@dataclass(frozen=True)
class GenerationProgress:
    """Latest structured per-step progress reported by the generator."""

    step: int
    completed_steps: int
    total_steps: int
    elapsed_s: float
    step_duration_s: float


@dataclass(frozen=True)
class GenerationResult:
    """Terminal, immutable result of a generation job."""

    state: GenerationState
    launched_revision: int
    current_revision: int
    stale: bool
    returncode: int | None
    paths: GenerationPaths
    progress: GenerationProgress | None
    events: tuple[Mapping[str, Any], ...]
    stdout_log: tuple[str, ...]
    stderr_log: tuple[str, ...]
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        """Return whether the generator published a new canonical frame set."""
        return self.state is GenerationState.SUCCEEDED


class JsonlEventDecoder:
    """Incrementally decode versioned JSONL records from arbitrary text chunks."""

    def __init__(self) -> None:
        self._pending = ""
        self.errors: list[str] = []
        self.lines: list[str] = []

    def feed(self, chunk: str) -> list[dict[str, Any]]:
        """Consume one possibly partial chunk and return complete valid events."""
        self._pending += chunk
        lines = self._pending.split("\n")
        self._pending = lines.pop()
        return self._decode_lines(lines)

    def finish(self) -> list[dict[str, Any]]:
        """Decode a final non-newline-terminated record, if present."""
        final = self._pending
        self._pending = ""
        return self._decode_lines([final] if final else [])

    def _decode_lines(self, lines: Sequence[str]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for raw_line in lines:
            line = raw_line.rstrip("\r")
            if not line:
                continue
            self.lines.append(line)
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                self.errors.append(f"invalid JSONL progress record: {exc.msg}: {line}")
                continue
            if not isinstance(value, dict):
                self.errors.append(f"progress record is not an object: {line}")
                continue
            if value.get("schema_version") != _PROGRESS_SCHEMA_VERSION:
                self.errors.append(
                    "unsupported progress schema_version: " f"{value.get('schema_version')!r}"
                )
                continue
            event_name = value.get("event")
            if event_name not in _PROGRESS_EVENTS:
                self.errors.append(f"unsupported progress event: {event_name!r}")
                continue
            events.append(value)
        return events


class _LineCollector:
    """Split diagnostic text chunks without losing a final partial line."""

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, chunk: str) -> list[str]:
        self._pending += chunk
        lines = self._pending.split("\n")
        self._pending = lines.pop()
        return [line.rstrip("\r") for line in lines]

    def finish(self) -> list[str]:
        if not self._pending:
            return []
        line = self._pending.rstrip("\r")
        self._pending = ""
        return [line]


class GenerationJob:
    """Generate one saved authoring revision without risking existing frames.

    Call ``start()`` once, optionally inspect ``progress`` and logs while it is
    running, then call ``wait()``. Callbacks run on background reader/waiter
    threads; Qt clients should marshal callback work onto the GUI thread.
    """

    def __init__(
        self,
        scenario_yaml: str | Path,
        launched_revision: int,
        *,
        command: Sequence[str] | None = None,
        revision_provider: Callable[[], int] | None = None,
        on_progress: Callable[[GenerationProgress], None] | None = None,
        on_log: Callable[[str, str], None] | None = None,
        on_finished: Callable[[GenerationResult], None] | None = None,
        cancel_grace_s: float = 2.0,
        job_id: str | None = None,
    ) -> None:
        source = Path(scenario_yaml).expanduser().resolve()
        if source.name != "scenario.yaml":
            raise ValueError("generation requires <scenario-directory>/scenario.yaml")
        if command is not None and not command:
            raise ValueError("generator command must not be empty")
        if cancel_grace_s < 0:
            raise ValueError("cancel_grace_s must be nonnegative")

        token = job_id or compact_uuid_token()
        if not _JOB_ID_RE.fullmatch(token):
            raise ValueError("job_id may contain only letters, digits, '_' and '-'")

        scenario_dir = source.parent
        source_bytes = source.read_bytes()
        source_text = source_bytes.decode("utf-8")
        configuration = self._read_configuration_text(source_text)
        validated_configuration = validate_scenario_data(configuration)
        if validated_configuration.data.mode != "files":
            raise ValueError("Builder Generate supports only file-mode HDF5 frames")
        files = validated_configuration.data.files
        if files.format not in {"h5", "hdf5"}:
            raise ValueError("Builder Generate supports only HDF5 frame output")
        if files.directory != DEFAULT_FRAMES_DIRECTORY:
            raise ValueError(
                "Builder Generate publishes only to <scenario>/frames; "
                "data.files.directory is a read-only frame selection"
            )
        raytracing_writes_frames = bool(validated_configuration.raytracing.enabled)
        coverage_config = validated_configuration.coverage
        coverage_writes_frames = bool(
            coverage_config is not None
            and coverage_config.enabled
            and (
                coverage_config.save is None
                or coverage_config.save.data is None
                or coverage_config.save.data.enabled
            )
        )
        if not raytracing_writes_frames and not coverage_writes_frames:
            raise ValueError(
                "Builder Generate requires a frame-producing ray-tracing or coverage "
                "run; use orchav-generator for summary-only or other derived-only "
                "workflows"
            )

        self.source_identity = loaded_source_identity("visualizer")
        self.paths = GenerationPaths(
            scenario_yaml=source,
            snapshot_yaml=scenario_dir / f"{_SNAPSHOT_PREFIX}{token}.yaml",
            final_frames=(scenario_dir / DEFAULT_FRAMES_DIRECTORY).resolve(),
        )
        _preflight_generation_paths(self.paths)
        self._snapshot_bytes = source_bytes
        self._configuration = configuration
        self._previous_frame_set_id = self._existing_frame_set_id(self.paths.final_frames)
        self.launched_revision = int(launched_revision)
        self.command = (
            tuple(str(part) for part in command)
            if command is not None
            else source_bound_module_command(
                "generator",
                identity=self.source_identity,
                anchor_package="visualizer",
            )
        )
        self.revision_provider = revision_provider
        self.on_progress = on_progress
        self.on_log = on_log
        self.on_finished = on_finished
        self.cancel_grace_s = float(cancel_grace_s)

        self._state = GenerationState.PENDING
        self._process: subprocess.Popen[str] | None = None
        self._progress: GenerationProgress | None = None
        self._events: list[dict[str, Any]] = []
        self._stdout_log: list[str] = []
        self._stderr_log: list[str] = []
        self._protocol_errors: list[str] = []
        self._saw_completed = False
        self._saw_started = False
        self._reported_failure: str | None = None
        self._cancel_requested = False
        self._result: GenerationResult | None = None
        self._snapshot_owned = False
        self._lock = threading.RLock()
        self._done = threading.Event()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._waiter_thread: threading.Thread | None = None

    @staticmethod
    def _read_configuration_text(source_text: str) -> dict[str, Any]:
        raw = yaml.safe_load(source_text)
        if not isinstance(raw, dict):
            raise ValueError("scenario.yaml must contain a YAML mapping")
        return raw

    @staticmethod
    def _existing_frame_set_id(frames_dir: Path) -> str | None:
        """Return a valid prior identity without duplicating writer safety checks."""

        try:
            return cast(
                str,
                load_frame_manifest(frames_dir, verify_files=False).frame_set_id,
            )
        except FrameManifestError:
            # The generator's shared writer remains the authority for deciding
            # whether an existing destination is replaceable. Builder needs a
            # prior identity only to reject a false successful no-op.
            return None

    @property
    def state(self) -> GenerationState:
        with self._lock:
            return self._state

    @property
    def process(self) -> subprocess.Popen[str] | None:
        """Return the owned subprocess, including after it exits."""
        with self._lock:
            return self._process

    @property
    def progress(self) -> GenerationProgress | None:
        with self._lock:
            return self._progress

    @property
    def stdout_log(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._stdout_log)

    @property
    def stderr_log(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._stderr_log)

    @property
    def result(self) -> GenerationResult | None:
        with self._lock:
            return self._result

    def is_stale(self, current_revision: int) -> bool:
        """Return whether the document has changed since this job launched."""
        return int(current_revision) != self.launched_revision

    def start(self) -> GenerationJob:
        """Create the immutable snapshot and launch the generator asynchronously."""
        with self._lock:
            if self._state is not GenerationState.PENDING:
                raise RuntimeError("a generation job can only be started once")

        try:
            self._write_snapshot()
            process = self._launch_process()
        except BaseException as exc:
            self._finish_without_process(f"could not launch generator: {exc}")
            if not isinstance(exc, Exception):
                raise
            return self

        with self._lock:
            self._process = process
            self._state = GenerationState.RUNNING

        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(process.stdout,),
            name=f"orchav-generation-stdout-{self.paths.snapshot_yaml.stem}",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(process.stderr,),
            name=f"orchav-generation-stderr-{self.paths.snapshot_yaml.stem}",
            daemon=True,
        )
        self._waiter_thread = threading.Thread(
            target=self._wait_for_process,
            args=(process,),
            name=f"orchav-generation-waiter-{self.paths.snapshot_yaml.stem}",
            daemon=True,
        )
        started_threads: list[threading.Thread] = []
        try:
            self._stdout_thread.start()
            started_threads.append(self._stdout_thread)
            self._stderr_thread.start()
            started_threads.append(self._stderr_thread)
            self._waiter_thread.start()
        except BaseException as exc:
            self._stop_process_after_worker_start_failure(process, started_threads)
            self._finish_without_process(
                "could not start generator monitoring workers " f"({type(exc).__name__}): {exc}"
            )
            if not isinstance(exc, Exception):
                raise
        return self

    def _stop_process_after_worker_start_failure(
        self,
        process: subprocess.Popen[str],
        started_threads: Sequence[threading.Thread],
    ) -> None:
        """Best-effort stop/reap after partial background-worker startup."""

        try:
            if process.poll() is None:
                process.terminate()
        except BaseException as exc:
            self._append_log(
                "stderr",
                f"could not terminate generator after worker startup failed: {exc}",
            )
        try:
            process.wait(timeout=max(self.cancel_grace_s, 0.1))
        except BaseException:
            try:
                process.kill()
            except BaseException as exc:
                self._append_log(
                    "stderr",
                    f"could not kill generator after worker startup failed: {exc}",
                )
            try:
                process.wait(timeout=1.0)
            except BaseException as exc:
                self._append_log(
                    "stderr",
                    f"could not reap generator after worker startup failed: {exc}",
                )

        for stream in (process.stdout, process.stderr):
            if stream is None:
                continue
            try:
                stream.close()
            except BaseException as exc:
                self._append_log(
                    "stderr",
                    f"could not close generator pipe after worker startup failed: {exc}",
                )
        for thread in started_threads:
            try:
                thread.join(timeout=1.0)
            except BaseException as exc:
                self._append_log(
                    "stderr",
                    f"could not join generator worker after startup failed: {exc}",
                )

    def cancel(self) -> bool:
        """Request cancellation, escalating to kill after the grace period."""
        with self._lock:
            process = self._process
            if self._state is not GenerationState.RUNNING or process is None:
                return False
            if process.poll() is not None:
                return False
            self._cancel_requested = True

        try:
            process.terminate()
        except OSError as exc:
            self._append_log("stderr", f"could not terminate generator: {exc}")

        threading.Thread(
            target=self._kill_after_grace,
            args=(process,),
            name=f"orchav-generation-cancel-{self.paths.snapshot_yaml.stem}",
            daemon=True,
        ).start()
        return True

    def wait(self, timeout: float | None = None) -> GenerationResult:
        """Wait for a terminal result or raise ``TimeoutError``."""
        with self._lock:
            if self._state is GenerationState.PENDING:
                raise RuntimeError("generation job has not been started")
        if not self._done.wait(timeout):
            raise TimeoutError("generation job did not finish before the timeout")
        with self._lock:
            assert self._result is not None
            return self._result

    def _write_snapshot(self) -> None:
        source = self.paths.scenario_yaml
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.read_bytes() != self._snapshot_bytes:
            raise RuntimeError(
                "scenario.yaml changed after generation was prepared; "
                "create a new generation job"
            )
        if self.paths.snapshot_yaml.exists():
            raise FileExistsError("generation snapshot path already exists")

        with self.paths.snapshot_yaml.open("xb") as handle:
            self._snapshot_owned = True
            handle.write(self._snapshot_bytes)

    def _launch_process(self) -> subprocess.Popen[str]:
        environment = os.environ.copy()
        environment.setdefault("PYTHONUNBUFFERED", "1")
        environment[EXPECTED_SOURCE_IDENTITY_ENV] = json.dumps(
            self.source_identity.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        command = [
            *self.command,
            "--progress-format",
            "jsonl",
            str(self.paths.scenario_yaml.parent),
            "--_authoring-snapshot-yaml",
            str(self.paths.snapshot_yaml),
        ]
        return subprocess.Popen(
            command,
            cwd=self.paths.scenario_yaml.parent,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

    def _read_stdout(self, stream: TextIO | None) -> None:
        if stream is None:
            with self._lock:
                self._protocol_errors.append("generator stdout pipe was unavailable")
            return
        decoder = JsonlEventDecoder()
        recorded_line_count = 0
        try:
            for line_chunk in stream:
                events = decoder.feed(line_chunk)
                for line in decoder.lines[recorded_line_count:]:
                    self._append_log("stdout", line)
                recorded_line_count = len(decoder.lines)
                for event in events:
                    self._record_event(event)
            events = decoder.finish()
            for line in decoder.lines[recorded_line_count:]:
                self._append_log("stdout", line)
            for event in events:
                self._record_event(event)
        except OSError as exc:
            decoder.errors.append(f"could not read generator stdout: {exc}")
        finally:
            with self._lock:
                self._protocol_errors.extend(decoder.errors)

    def _read_stderr(self, stream: TextIO | None) -> None:
        if stream is None:
            return
        collector = _LineCollector()
        try:
            for line_chunk in stream:
                for line in collector.feed(line_chunk):
                    self._append_log("stderr", line)
            for line in collector.finish():
                self._append_log("stderr", line)
        except OSError as exc:
            self._append_log("stderr", f"could not read generator stderr: {exc}")

    def _record_event(self, event: dict[str, Any]) -> None:
        progress_callback: Callable[[GenerationProgress], None] | None = None
        progress_value: GenerationProgress | None = None
        with self._lock:
            self._events.append(dict(event))
            event_name = event["event"]
            if event_name == "run_started":
                if self._saw_started:
                    self._protocol_errors.append("generator reported run_started more than once")
                self._saw_started = True
                try:
                    child_identity = SourceIdentity.from_mapping(event.get("source_identity"))
                except (TypeError, ValueError) as exc:
                    self._protocol_errors.append(f"invalid generator source identity: {exc}")
                else:
                    if not self.source_identity.matches(child_identity):
                        self._protocol_errors.append(
                            "generator source identity does not match the running visualizer: "
                            f"expected {self.source_identity.to_dict()!r}, "
                            f"got {child_identity.to_dict()!r}"
                        )
            elif event_name == "step_completed":
                try:
                    progress_value = GenerationProgress(
                        step=int(event["step"]),
                        completed_steps=int(event["completed_steps"]),
                        total_steps=int(event["total_steps"]),
                        elapsed_s=float(event["elapsed_s"]),
                        step_duration_s=float(event["step_duration_s"]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    self._protocol_errors.append(f"invalid step_completed event: {exc}")
                else:
                    self._progress = progress_value
                    progress_callback = self.on_progress
            elif event_name == "run_completed":
                self._saw_completed = True
            elif event_name == "run_failed":
                self._reported_failure = str(event.get("message") or "generator reported failure")

        if progress_callback is not None and progress_value is not None:
            self._invoke_callback(progress_callback, progress_value)

    def _append_log(self, stream_name: str, line: str) -> None:
        with self._lock:
            target = self._stdout_log if stream_name == "stdout" else self._stderr_log
            target.append(line)
        self._notify_log(stream_name, line)

    def _notify_log(self, stream_name: str, line: str) -> None:
        callback = self.on_log
        if callback is not None:
            self._invoke_callback(callback, stream_name, line)

    @staticmethod
    def _invoke_callback(callback: Callable[..., Any], *args: Any) -> None:
        try:
            callback(*args)
        except BaseException:
            logger.exception("Generation job callback failed")

    def _wait_for_process(self, process: subprocess.Popen[str]) -> None:
        try:
            returncode = process.wait()
            if self._stdout_thread is not None:
                self._stdout_thread.join()
            if self._stderr_thread is not None:
                self._stderr_thread.join()
            self._finish_process(returncode)
        except BaseException as exc:
            with self._lock:
                already_terminal = self._result is not None
            if already_terminal:
                return
            readers = tuple(
                thread
                for thread in (self._stdout_thread, self._stderr_thread)
                if thread is not None and thread.ident is not None
            )
            self._stop_process_after_worker_start_failure(process, readers)
            self._finish_terminal(
                GenerationState.FAILED,
                returncode=process.poll(),
                error_message=("generator monitoring failed " f"({type(exc).__name__}): {exc}"),
            )

    def _kill_after_grace(self, process: subprocess.Popen[str]) -> None:
        try:
            process.wait(timeout=self.cancel_grace_s)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError as exc:
                self._append_log("stderr", f"could not kill generator: {exc}")

    def _finish_process(self, returncode: int) -> None:
        with self._lock:
            cancelled = self._cancel_requested
            protocol_errors = tuple(self._protocol_errors)
            reported_failure = self._reported_failure
            saw_completed = self._saw_completed
            saw_started = self._saw_started

        state: GenerationState
        error: str | None = None
        if cancelled:
            state = GenerationState.CANCELLED
            error = "generation was cancelled"
        elif returncode != 0:
            state = GenerationState.FAILED
            error = reported_failure or f"generator exited with status {returncode}"
        elif reported_failure:
            state = GenerationState.FAILED
            error = reported_failure
        elif protocol_errors:
            state = GenerationState.FAILED
            error = "; ".join(protocol_errors)
        elif not saw_started:
            state = GenerationState.FAILED
            error = "generator exited without a run_started progress event"
        elif not saw_completed:
            state = GenerationState.FAILED
            error = "generator exited without a run_completed progress event"
        else:
            try:
                self._verify_direct_publication()
            except BaseException as exc:
                state = GenerationState.FAILED
                error = (
                    "could not verify directly published generation output "
                    f"({type(exc).__name__}): {exc}"
                )
            else:
                state = GenerationState.SUCCEEDED

        self._finish_terminal(
            state,
            returncode=returncode,
            error_message=error,
        )

    def _finish_without_process(self, error_message: str) -> None:
        self._finish_terminal(
            GenerationState.FAILED,
            returncode=None,
            error_message=error_message,
        )

    def _finish_terminal(
        self,
        state: GenerationState,
        *,
        returncode: int | None,
        error_message: str | None,
    ) -> None:
        """Always publish a terminal result after attempting owned cleanup."""

        try:
            self._cleanup_working_paths()
        except BaseException as exc:
            self._append_log(
                "stderr",
                "generation working-path cleanup was interrupted " f"({type(exc).__name__}: {exc})",
            )
        finally:
            self._publish_result(
                state,
                returncode=returncode,
                error_message=error_message,
            )

    def _verify_direct_publication(self) -> None:
        """Require a complete canonical frame manifest with a new identity."""

        manifest = load_frame_manifest(self.paths.final_frames)
        if not manifest.frame_ids or not manifest.chunks:
            raise ValueError("generator published an empty frame set")
        if manifest.frame_set_id == self._previous_frame_set_id:
            raise ValueError(
                "generator reported success without publishing a new frame-set identity"
            )

    def _cleanup_working_paths(self) -> None:
        snapshot = self.paths.snapshot_yaml
        scenario_dir = self.paths.scenario_yaml.parent

        if (
            self._snapshot_owned
            and snapshot.parent == scenario_dir
            and snapshot.name.startswith(_SNAPSHOT_PREFIX)
        ):
            try:
                snapshot.unlink(missing_ok=True)
                self._snapshot_owned = False
            except OSError as exc:
                self._append_log("stderr", f"could not remove generation snapshot: {exc}")

    def _current_revision(self) -> int:
        provider = self.revision_provider
        if provider is None:
            return self.launched_revision
        try:
            return int(provider())
        except BaseException as exc:
            self._append_log(
                "stderr",
                "could not read current document revision " f"({type(exc).__name__}): {exc}",
            )
            return self.launched_revision

    def _publish_result(
        self,
        state: GenerationState,
        *,
        returncode: int | None,
        error_message: str | None,
    ) -> None:
        current_revision = self._current_revision()
        with self._lock:
            self._state = state
            result = GenerationResult(
                state=state,
                launched_revision=self.launched_revision,
                current_revision=current_revision,
                stale=self.is_stale(current_revision),
                returncode=returncode,
                paths=self.paths,
                progress=self._progress,
                events=tuple(dict(event) for event in self._events),
                stdout_log=tuple(self._stdout_log),
                stderr_log=tuple(self._stderr_log),
                error_message=error_message,
            )
            self._result = result
            self._done.set()

        if self.on_finished is not None:
            self._invoke_callback(self.on_finished, result)


__all__ = [
    "GenerationJob",
    "GenerationPaths",
    "GenerationProgress",
    "GenerationResult",
    "GenerationState",
    "JsonlEventDecoder",
]
