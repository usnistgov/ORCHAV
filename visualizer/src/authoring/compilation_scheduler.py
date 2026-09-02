"""Latest-only background compilation for the Scenario Builder.

This module is independent of the workspace. It accepts an immutable domain
snapshot at the document boundary and marshals compiler
outcomes back to the scheduler's Qt owner thread.
"""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from PySide6.QtCore import QObject, Qt, Signal, Slot

from .compiler import CompilationResult, ScenarioCompiler
from .domain import AuthoringScenario


class _Compiler(Protocol):
    def compile(
        self,
        scenario: AuthoringScenario,
        *,
        scenario_directory: Path | None = None,
    ) -> CompilationResult: ...


_CompilerFactory = Callable[[], _Compiler]


@dataclass(frozen=True, slots=True)
class CompilationFailure:
    """Thread-neutral description of an unexpected compiler exception."""

    exception_type: str
    message: str

    @classmethod
    def from_exception(cls, exception: Exception) -> "CompilationFailure":
        """Copy useful exception details without retaining a worker traceback."""

        exception_type = f"{type(exception).__module__}.{type(exception).__qualname__}"
        return cls(exception_type=exception_type, message=str(exception))


@dataclass(frozen=True, slots=True)
class _CompilationRequest:
    scenario: AuthoringScenario
    scenario_directory: Path
    token: int


@dataclass(frozen=True, slots=True)
class _CompilationOutcome:
    token: int
    result: CompilationResult | None = None
    failure: CompilationFailure | None = None


class AuthoringCompilationScheduler(QObject):
    """Compile immutable authoring snapshots without blocking the Qt thread.

    At most one compile runs at a time.  While it runs, repeated requests
    replace one pending slot, so only the newest snapshot is compiled next.
    Results pass through both worker-side and owner-thread token checks; an
    outcome that became stale while queued in Qt is never published.

    ``close()`` is non-blocking because generator-backed compilation is not
    generally interruptible.  The persistent worker is a daemon thread and
    suppresses all outcomes after close, making Qt owner teardown safe while a
    compile winds down naturally.  The compiler factory runs on that worker
    once, preserving its private scene and target caches for scheduler life.
    """

    succeeded = Signal(object, object)
    failed = Signal(object, object)

    _outcome_ready = Signal(object)

    def __init__(
        self,
        compiler_factory: _CompilerFactory = ScenarioCompiler,
        *,
        compile_lock: AbstractContextManager[object] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if not callable(compiler_factory):
            raise TypeError("compiler_factory must be callable")
        self._compiler_factory = compiler_factory
        self._compile_lock = compile_lock if compile_lock is not None else threading.Lock()
        self._state_lock = threading.Lock()
        self._state_condition = threading.Condition(self._state_lock)
        self._pending: _CompilationRequest | None = None
        self._latest_token: int | None = None
        self._worker_running = False
        self._worker_started = False
        self._closed = False
        self._outcome_ready.connect(
            self._deliver_outcome,
            Qt.ConnectionType.QueuedConnection,
        )

    @property
    def closed(self) -> bool:
        """Return whether this scheduler permanently stopped accepting work."""

        with self._state_lock:
            return self._closed

    @property
    def running(self) -> bool:
        """Return whether a worker owns or is about to own a request."""

        with self._state_lock:
            return self._worker_running

    @property
    def latest_request_token(self) -> int | None:
        """Return the newest accepted caller token."""

        with self._state_lock:
            return self._latest_token

    def request(
        self,
        scenario: AuthoringScenario,
        scenario_directory: Path,
        token: int,
    ) -> bool:
        """Queue a snapshot and return immediately.

        Tokens must be strictly increasing for the lifetime of the scheduler.
        Returning ``False`` means the scheduler was already closed. Any pending
        request is atomically replaced by this one.
        """

        if not isinstance(scenario, AuthoringScenario):
            raise TypeError("scenario must be an immutable AuthoringScenario snapshot")
        self._validate_token(token)

        request = _CompilationRequest(
            scenario=scenario,
            scenario_directory=Path(scenario_directory),
            token=token,
        )
        should_start = False
        with self._state_condition:
            if self._closed:
                return False
            self._require_newer_token_locked(token)
            self._latest_token = token
            self._pending = request
            self._worker_running = True
            if not self._worker_started:
                self._worker_started = True
                should_start = True
            self._state_condition.notify()

        if should_start:
            threading.Thread(
                target=self._run_worker,
                name="orchav-authoring-compiler",
                daemon=True,
            ).start()
        return True

    def invalidate(self, token: int) -> bool:
        """Drop pending work and invalidate active outcomes with lower tokens.

        The caller supplies the next token from the same strictly increasing
        sequence used by :meth:`request`.  This is the synchronous-compilation
        handoff: invalidate first, then compile that immutable snapshot with a
        separate compiler. Any completion with a lower token is discarded even
        if its Qt delivery is already queued.

        Returning ``False`` means the scheduler was already closed.
        """

        self._validate_token(token)
        with self._state_condition:
            if self._closed:
                return False
            self._require_newer_token_locked(token)
            self._latest_token = token
            self._pending = None
        return True

    def close(self) -> None:
        """Reject requests, drop pending work, and ignore the active outcome."""

        with self._state_condition:
            self._closed = True
            self._pending = None
            self._state_condition.notify_all()

    def _run_worker(self) -> None:
        compiler: _Compiler | None = None
        while True:
            with self._state_condition:
                while not self._closed and self._pending is None:
                    self._worker_running = False
                    self._state_condition.wait()
                if self._closed:
                    self._worker_running = False
                    self._worker_started = False
                    return
                request = self._pending
                self._pending = None
                self._worker_running = True
            if request is None:
                continue

            try:
                if compiler is None:
                    compiler = self._compiler_factory()
                with self._compile_lock:
                    result = compiler.compile(
                        request.scenario,
                        scenario_directory=request.scenario_directory,
                    )
            except Exception as exception:  # noqa: BLE001 - worker boundary
                outcome = _CompilationOutcome(
                    token=request.token,
                    failure=CompilationFailure.from_exception(exception),
                )
            else:
                outcome = _CompilationOutcome(token=request.token, result=result)

            with self._state_lock:
                should_offer = not self._closed and request.token == self._latest_token
            if should_offer:
                try:
                    self._outcome_ready.emit(outcome)
                except RuntimeError:
                    # The QObject may have been deleted just after close won
                    # the state race.  There is no receiver left to notify.
                    pass

    @staticmethod
    def _validate_token(token: int) -> None:
        if isinstance(token, bool) or not isinstance(token, int):
            raise TypeError("request token must be an integer")

    def _require_newer_token_locked(self, token: int) -> None:
        if self._latest_token is not None and token <= self._latest_token:
            raise ValueError(
                f"request token must be greater than the previous token ({self._latest_token})"
            )

    @Slot(object)
    def _deliver_outcome(self, outcome: _CompilationOutcome) -> None:
        with self._state_lock:
            publish = not self._closed and outcome.token == self._latest_token
        if not publish:
            return
        if outcome.failure is not None:
            self.failed.emit(outcome.token, outcome.failure)
        elif outcome.result is not None:
            self.succeeded.emit(outcome.token, outcome.result)


__all__ = ["AuthoringCompilationScheduler", "CompilationFailure"]
