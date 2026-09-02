"""Deterministic Qt/thread tests for latest-only authoring compilation."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from visualizer.src.authoring.compilation_scheduler import (
    AuthoringCompilationScheduler,
    CompilationFailure,
)
from visualizer.src.authoring.domain import AuthoringScenario, SceneReference

SCENARIO_DIRECTORY = Path(__file__).resolve().parent


def _scenario(name: str) -> AuthoringScenario:
    return AuthoringScenario(scene=SceneReference(source="library", id=name))


def _wait_without_qt(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for worker state")
        time.sleep(0.005)


class _BlockingCompiler:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.thread_id: int | None = None

    def compile(self, scenario, *, scenario_directory=None):
        self.thread_id = threading.get_ident()
        self.started.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("test did not release the blocking compiler")
        self.finished.set()
        return scenario.scene.id


class _FirstCallBlockingCompiler:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.latest_finished = threading.Event()

    def compile(self, scenario, *, scenario_directory=None):
        name = scenario.scene.id
        self.calls.append(name)
        if len(self.calls) == 1:
            self.first_started.set()
            if not self.release_first.wait(timeout=2.0):
                raise TimeoutError("test did not release the first compile")
        if name == "third":
            self.latest_finished.set()
        return name


class _SecondCallBlockingCompiler:
    def __init__(self) -> None:
        self.calls = 0
        self.first_finished = threading.Event()
        self.second_started = threading.Event()
        self.release_second = threading.Event()

    def compile(self, scenario, *, scenario_directory=None):
        self.calls += 1
        if self.calls == 1:
            self.first_finished.set()
            return scenario.scene.id
        self.second_started.set()
        if not self.release_second.wait(timeout=2.0):
            raise TimeoutError("test did not release the second compile")
        return scenario.scene.id


def test_request_returns_immediately_and_delivers_on_owner_thread(qapp, qtbot) -> None:
    compiler = _BlockingCompiler()
    factory_thread_ids: list[int] = []

    def compiler_factory():
        factory_thread_ids.append(threading.get_ident())
        return compiler

    scheduler = AuthoringCompilationScheduler(compiler_factory)
    owner_thread_id = threading.get_ident()
    delivered: list[tuple[int, str, int]] = []
    scheduler.succeeded.connect(
        lambda token, result: delivered.append((token, result, threading.get_ident()))
    )

    started_at = time.monotonic()
    assert scheduler.request(_scenario("first"), SCENARIO_DIRECTORY, 1) is True
    elapsed = time.monotonic() - started_at

    assert elapsed < 1.0
    assert compiler.started.wait(timeout=1.0)
    assert factory_thread_ids == [compiler.thread_id]
    assert compiler.thread_id != owner_thread_id

    compiler.release.set()
    qtbot.waitUntil(lambda: len(delivered) == 1, timeout=2000)
    assert delivered == [(1, "first", owner_thread_id)]
    _wait_without_qt(lambda: not scheduler.running)

    assert scheduler.request(_scenario("second"), SCENARIO_DIRECTORY, 2) is True
    qtbot.waitUntil(lambda: len(delivered) == 2, timeout=2000)
    assert delivered[-1] == (2, "second", owner_thread_id)
    assert len(factory_thread_ids) == 1
    scheduler.close()


def test_compile_lock_is_held_for_the_entire_worker_compile(qapp) -> None:
    compiler = _BlockingCompiler()
    compile_lock = threading.Lock()
    scheduler = AuthoringCompilationScheduler(
        lambda: compiler,
        compile_lock=compile_lock,
    )

    assert scheduler.request(_scenario("locked"), SCENARIO_DIRECTORY, 5) is True
    assert compiler.started.wait(timeout=1.0)
    assert compile_lock.acquire(blocking=False) is False

    compiler.release.set()
    assert compiler.finished.wait(timeout=1.0)
    _wait_without_qt(lambda: not scheduler.running)
    assert compile_lock.acquire(blocking=False) is True
    compile_lock.release()
    scheduler.close()


def test_pending_requests_coalesce_and_only_newest_result_is_published(qapp, qtbot) -> None:
    compiler = _FirstCallBlockingCompiler()
    scheduler = AuthoringCompilationScheduler(lambda: compiler)
    delivered: list[tuple[int, str]] = []
    scheduler.succeeded.connect(lambda token, result: delivered.append((token, result)))

    assert scheduler.request(_scenario("first"), SCENARIO_DIRECTORY, 10) is True
    assert compiler.first_started.wait(timeout=1.0)
    assert scheduler.request(_scenario("second"), SCENARIO_DIRECTORY, 11) is True
    assert scheduler.request(_scenario("third"), SCENARIO_DIRECTORY, 12) is True
    compiler.release_first.set()

    assert compiler.latest_finished.wait(timeout=1.0)
    qtbot.waitUntil(lambda: delivered == [(12, "third")], timeout=2000)
    assert compiler.calls == ["first", "third"]
    scheduler.close()


def test_outcome_that_becomes_stale_in_qt_queue_is_not_published(qapp, qtbot) -> None:
    compiler = _SecondCallBlockingCompiler()
    scheduler = AuthoringCompilationScheduler(lambda: compiler)
    delivered: list[tuple[int, str]] = []
    scheduler.succeeded.connect(lambda token, result: delivered.append((token, result)))

    assert scheduler.request(_scenario("first"), SCENARIO_DIRECTORY, 20) is True
    assert compiler.first_finished.wait(timeout=1.0)
    _wait_without_qt(lambda: not scheduler.running)

    # The first outcome is now queued to Qt but deliberately not processed.
    assert scheduler.request(_scenario("second"), SCENARIO_DIRECTORY, 21) is True
    assert compiler.second_started.wait(timeout=1.0)
    qapp.processEvents()
    assert delivered == []

    compiler.release_second.set()
    qtbot.waitUntil(lambda: delivered == [(21, "second")], timeout=2000)
    scheduler.close()


def test_invalidate_drops_pending_and_suppresses_active_result(qapp, qtbot) -> None:
    compiler = _FirstCallBlockingCompiler()
    scheduler = AuthoringCompilationScheduler(lambda: compiler)
    delivered: list[tuple[int, str]] = []
    scheduler.succeeded.connect(lambda token, result: delivered.append((token, result)))

    assert scheduler.request(_scenario("active"), SCENARIO_DIRECTORY, 25) is True
    assert compiler.first_started.wait(timeout=1.0)
    assert scheduler.request(_scenario("pending"), SCENARIO_DIRECTORY, 26) is True

    # Token 27 belongs to a synchronous compile owned by the caller.
    assert scheduler.invalidate(27) is True
    assert scheduler.latest_request_token == 27
    compiler.release_first.set()
    _wait_without_qt(lambda: not scheduler.running)
    qapp.processEvents()

    assert compiler.calls == ["active"]
    assert delivered == []

    assert scheduler.request(_scenario("after-sync"), SCENARIO_DIRECTORY, 28) is True
    qtbot.waitUntil(lambda: delivered == [(28, "after-sync")], timeout=2000)
    assert compiler.calls == ["active", "after-sync"]
    scheduler.close()


def test_latest_failure_is_delivered_on_owner_thread(qapp, qtbot) -> None:
    class _FailingCompiler:
        def compile(self, scenario, *, scenario_directory=None):
            raise RuntimeError("compile exploded")

    scheduler = AuthoringCompilationScheduler(_FailingCompiler)
    owner_thread_id = threading.get_ident()
    failures: list[tuple[int, CompilationFailure, int]] = []
    successes: list[object] = []
    scheduler.failed.connect(
        lambda token, failure: failures.append((token, failure, threading.get_ident()))
    )
    scheduler.succeeded.connect(lambda _token, result: successes.append(result))

    assert scheduler.request(_scenario("broken"), SCENARIO_DIRECTORY, 30) is True
    qtbot.waitUntil(lambda: len(failures) == 1, timeout=2000)

    token, failure, delivery_thread_id = failures[0]
    assert token == 30
    assert failure == CompilationFailure("builtins.RuntimeError", "compile exploded")
    assert delivery_thread_id == owner_thread_id
    assert successes == []
    scheduler.close()


def test_close_drops_pending_and_active_outcomes_safely(qapp) -> None:
    compiler = _BlockingCompiler()
    scheduler = AuthoringCompilationScheduler(lambda: compiler)
    delivered: list[object] = []
    scheduler.succeeded.connect(lambda _token, result: delivered.append(result))
    scheduler.failed.connect(lambda _token, failure: delivered.append(failure))

    assert scheduler.request(_scenario("active"), SCENARIO_DIRECTORY, 40) is True
    assert compiler.started.wait(timeout=1.0)
    scheduler.close()
    scheduler.close()

    assert scheduler.closed is True
    assert scheduler.request(_scenario("ignored"), SCENARIO_DIRECTORY, 41) is False
    compiler.release.set()
    assert compiler.finished.wait(timeout=1.0)
    _wait_without_qt(lambda: not scheduler.running)
    qapp.processEvents()
    assert delivered == []


def test_request_tokens_must_be_strictly_increasing(qapp) -> None:
    compiler = _BlockingCompiler()
    scheduler = AuthoringCompilationScheduler(lambda: compiler)
    assert scheduler.request(_scenario("first"), SCENARIO_DIRECTORY, 50) is True

    with pytest.raises(ValueError, match="greater than the previous token"):
        scheduler.request(_scenario("duplicate"), SCENARIO_DIRECTORY, 50)

    scheduler.close()
    compiler.release.set()
    assert compiler.finished.wait(timeout=1.0)
